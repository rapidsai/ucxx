from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

import dask
from distributed import Client, Scheduler, wait
from distributed.comm import connect, listen, parse_address
from distributed.comm.core import CommClosedError
from distributed.comm.registry import get_backend
from distributed.deploy.local import LocalCluster
from distributed.diagnostics.nvml import (
    device_get_count,
    get_device_index_and_uuid,
    get_device_mig_mode,
    has_cuda_context,
)
from distributed.protocol import to_serialize
from distributed.utils import wait_for
from distributed.utils_test import inc

import ucxx

import distributed_ucxx  # noqa: E402
from distributed_ucxx.utils_test import gen_test

try:
    HOST = ucxx.get_address()
except Exception:
    HOST = "127.0.0.1"


def test_registered(ucxx_loop):
    backend = get_backend("ucxx")
    assert isinstance(backend, distributed_ucxx.UCXXBackend)


async def get_comm_pair(
    listen_addr=f"ucxx://{HOST}", listen_args=None, connect_args=None, **kwargs
):
    listen_args = listen_args or {}
    connect_args = connect_args or {}
    q = asyncio.queues.Queue()

    async def handle_comm(comm):
        await q.put(comm)

    listener = listen(listen_addr, handle_comm, **listen_args, **kwargs)
    async with listener:
        comm = await connect(listener.contact_address, **connect_args, **kwargs)
        serv_comm = await q.get()
        return (comm, serv_comm)


@gen_test()
async def test_ping_pong(ucxx_loop):
    com, serv_com = await get_comm_pair()
    msg = {"op": "ping"}
    await com.write(msg)
    result = await serv_com.read()
    assert result == msg
    result["op"] = "pong"

    await serv_com.write(result)

    result = await com.read()
    assert result == {"op": "pong"}

    await com.close()
    await serv_com.close()


@gen_test()
async def test_comm_objs(ucxx_loop):
    comm, serv_comm = await get_comm_pair()

    scheme, loc = parse_address(comm.peer_address)
    assert scheme == "ucxx"

    scheme, loc = parse_address(serv_comm.peer_address)
    assert scheme == "ucxx"

    assert comm.peer_address == serv_comm.local_address


@gen_test()
async def test_ucxx_specific(ucxx_loop):
    """
    Test concrete UCXX API.
    """
    # TODO:
    # 1. ensure exceptions in handle_comm fail the test
    # 2. Use dict in read / write, put serialization there.
    # 3. Test peer_address
    # 4. Test cleanup
    address = f"ucxx://{HOST}:{0}"

    async def handle_comm(comm):
        msg = await comm.read()
        msg["op"] = "pong"
        await comm.write(msg)
        await comm.read()
        await comm.close()
        assert comm.closed() is True

    listener = await distributed_ucxx.UCXXListener(address, handle_comm)
    host, port = listener.get_host_port()
    assert host.count(".") == 3
    assert port > 0

    keys = []

    async def client_communicate(key, delay=0):
        # addr = "%s:%d" % (host, port)
        comm = await connect(listener.contact_address)
        # TODO: peer_address
        # assert comm.peer_address == 'ucxx://' + addr
        assert comm.extra_info == {}
        msg = {"op": "ping", "data": key}
        await comm.write(msg)
        if delay:
            await asyncio.sleep(delay)
        msg = await comm.read()
        assert msg == {"op": "pong", "data": key}
        await comm.write({"op": "client closed"})
        keys.append(key)
        return comm

    await client_communicate(key=1234, delay=0.5)

    # Many clients at once
    N = 2
    futures = [client_communicate(key=i, delay=0.05) for i in range(N)]
    await asyncio.gather(*futures)
    assert set(keys) == {1234} | set(range(N))

    listener.stop()


@gen_test()
async def test_ping_pong_data(ucxx_loop):
    np = pytest.importorskip("numpy")

    data = np.ones((10, 10))

    com, serv_com = await get_comm_pair()
    msg = {"op": "ping", "data": to_serialize(data)}
    await com.write(msg)
    result = await serv_com.read()
    result["op"] = "pong"
    data2 = result.pop("data")
    np.testing.assert_array_equal(data2, data)

    await serv_com.write(result)

    result = await com.read()
    assert result == {"op": "pong"}

    await com.close()
    await serv_com.close()


@pytest.mark.parametrize(
    "g",
    [
        lambda cudf: cudf.Series([1, 2, 3]),
        lambda cudf: cudf.Series([], dtype=object),
        lambda cudf: cudf.DataFrame([], dtype=object),
        lambda cudf: cudf.DataFrame([1]).head(0),
        lambda cudf: cudf.DataFrame([1.0]).head(0),
        lambda cudf: cudf.DataFrame({"a": []}),
        lambda cudf: cudf.DataFrame({"a": ["a"]}).head(0),
        lambda cudf: cudf.DataFrame({"a": [1.0]}).head(0),
        lambda cudf: cudf.DataFrame({"a": [1]}).head(0),
        lambda cudf: cudf.DataFrame({"a": [1, 2, None], "b": [1.0, 2.0, None]}),
        lambda cudf: cudf.DataFrame({"a": ["Check", "str"], "b": ["Sup", "port"]}),
    ],
)
@gen_test()
async def test_ping_pong_cudf(ucxx_loop, g):
    # if this test appears after cupy an import error arises
    # *** ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.11'
    # not found (required by python3.7/site-packages/pyarrow/../../../libarrow.so.12)
    cudf = pytest.importorskip("cudf")
    from cudf.testing import assert_eq

    cudf_obj = g(cudf)

    com, serv_com = await get_comm_pair()
    msg = {"op": "ping", "data": to_serialize(cudf_obj)}

    await com.write(msg)
    result = await serv_com.read()

    cudf_obj_2 = result.pop("data")
    assert result["op"] == "ping"
    assert_eq(cudf_obj, cudf_obj_2)

    await com.close()
    await serv_com.close()


@pytest.mark.parametrize("shape", [(100,), (10, 10), (4947,)])
@gen_test()
async def test_ping_pong_cupy(ucxx_loop, shape):
    cupy = pytest.importorskip("cupy")
    com, serv_com = await get_comm_pair()

    arr = cupy.random.random(shape)
    msg = {"op": "ping", "data": to_serialize(arr)}

    _, result = await asyncio.gather(com.write(msg), serv_com.read())
    data2 = result.pop("data")

    assert result["op"] == "ping"
    cupy.testing.assert_array_equal(arr, data2)
    await com.close()
    await serv_com.close()


@pytest.mark.slow
@pytest.mark.parametrize("n", [int(1e9), int(2.5e9)])
@gen_test()
async def test_large_cupy(ucxx_loop, n, cleanup):
    cupy = pytest.importorskip("cupy")
    com, serv_com = await get_comm_pair()

    arr = cupy.ones(n, dtype="u1")
    msg = {"op": "ping", "data": to_serialize(arr)}

    _, result = await asyncio.gather(com.write(msg), serv_com.read())
    data2 = result.pop("data")

    assert result["op"] == "ping"
    cupy.testing.assert_array_equal(data2, arr)
    await com.close()
    await serv_com.close()


@gen_test()
async def test_ping_pong_numba(ucxx_loop):
    np = pytest.importorskip("numpy")
    numba = pytest.importorskip("numba")
    import numba.cuda

    arr = np.arange(10)
    arr = numba.cuda.to_device(arr)

    com, serv_com = await get_comm_pair()
    msg = {"op": "ping", "data": to_serialize(arr)}

    await com.write(msg)
    result = await serv_com.read()
    data2 = result.pop("data")
    np.testing.assert_array_equal(data2, arr)
    assert result["op"] == "ping"


@pytest.mark.parametrize("processes", [True, False])
@pytest.mark.flaky(
    reruns=3,
    only_rerun="Trying to reset UCX but not all Endpoints and/or Listeners are closed",
)
@gen_test()
async def test_ucxx_localcluster(ucxx_loop, processes, cleanup):
    async with LocalCluster(
        protocol="ucxx",
        host=HOST,
        dashboard_address=":0",
        n_workers=2,
        threads_per_worker=1,
        processes=processes,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            x = client.submit(inc, 1)
            await x
            assert x.key in cluster.scheduler.tasks
            if not processes:
                assert any(w.data == {x.key: 2} for w in cluster.workers.values())
            assert len(cluster.scheduler.workers) == 2


@pytest.mark.slow
@gen_test(timeout=60)
async def test_stress(
    ucxx_loop,
):
    da = pytest.importorskip("dask.array")

    chunksize = "10 MB"

    async with LocalCluster(
        protocol="ucxx",
        dashboard_address=":0",
        asynchronous=True,
        host=HOST,
    ) as cluster:
        async with Client(cluster, asynchronous=True):
            rs = da.random.RandomState()
            x = rs.random((10000, 10000), chunks=(-1, chunksize))
            x = x.persist()
            await wait(x)

            for _ in range(10):
                x = x.rechunk((chunksize, -1))
                x = x.rechunk((-1, chunksize))
                x = x.persist()
                await wait(x)


@gen_test()
async def test_simple(
    ucxx_loop,
):
    async with LocalCluster(
        protocol="ucxx", n_workers=2, threads_per_worker=2, asynchronous=True
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            assert cluster.scheduler_address.startswith("ucxx://")
            assert await client.submit(lambda x: x + 1, 10) == 11


@pytest.mark.xfail(reason="If running on Docker, requires --pid=host")
@gen_test()
async def test_cuda_context(
    ucxx_loop,
):
    try:
        device_info = get_device_index_and_uuid(
            next(
                filter(
                    lambda i: get_device_mig_mode(i)[0] == 0, range(device_get_count())
                )
            )
        )
    except StopIteration:
        pytest.skip("No CUDA device in non-MIG mode available")

    with patch.dict(
        os.environ, {"CUDA_VISIBLE_DEVICES": device_info.uuid.decode("utf-8")}
    ):
        with dask.config.set({"distributed.comm.ucx.create-cuda-context": True}):
            async with LocalCluster(
                protocol="ucxx", n_workers=1, asynchronous=True
            ) as cluster:
                async with Client(cluster, asynchronous=True) as client:
                    assert cluster.scheduler_address.startswith("ucxx://")
                    ctx = has_cuda_context()
                    assert ctx.has_context and ctx.device_info == device_info
                    worker_cuda_context = await client.run(has_cuda_context)
                    assert len(worker_cuda_context) == 1
                    worker_cuda_context = list(worker_cuda_context.values())
                    assert (
                        worker_cuda_context[0].has_context
                        and worker_cuda_context[0].device_info == device_info
                    )


@pytest.mark.flaky(
    reruns=3,
    only_rerun="Trying to reset UCX but not all Endpoints and/or Listeners are closed",
)
@gen_test()
async def test_transpose(
    ucxx_loop,
):
    da = pytest.importorskip("dask.array")

    async with LocalCluster(
        protocol="ucxx", n_workers=2, threads_per_worker=2, asynchronous=True
    ) as cluster:
        async with Client(cluster, asynchronous=True):
            assert cluster.scheduler_address.startswith("ucxx://")
            x = da.ones((10000, 10000), chunks=(1000, 1000)).persist()
            await x
            y = (x + x.T).sum()
            await y


@pytest.mark.parametrize("port", [0, 1234])
@gen_test()
async def test_ucxx_protocol(ucxx_loop, cleanup, port):
    async with Scheduler(protocol="ucxx", port=port, dashboard_address=":0") as s:
        assert s.address.startswith("ucxx://")


@gen_test()
@pytest.mark.ignore_alive_references(True)
async def test_ucxx_unreachable(
    ucxx_loop,
):
    # It is not entirely clear why, but when attempting to reconnect
    # Distributed may fail to complete async tasks, leaving UCXX references
    # still alive. For now we disable those errors that only occur during the
    # teardown phase of this test.

    with pytest.raises(OSError, match="Timed out trying to connect to"):
        await Client("ucxx://255.255.255.255:12345", timeout=1, asynchronous=True)


@gen_test()
async def test_comm_closed_on_read_error():
    reader, writer = await get_comm_pair()

    # Depending on the UCP protocol selected, it may raise either
    # `asyncio.TimeoutError` or `CommClosedError`, so validate either one.
    with pytest.raises((asyncio.TimeoutError, CommClosedError)):
        await wait_for(reader.read(), 0.01)

    await writer.close()

    assert reader.closed()
    assert writer.closed()


@pytest.mark.flaky(
    reruns=3,
    only_rerun="Trying to reset UCX but not all Endpoints and/or Listeners are closed",
)
@gen_test()
async def test_embedded_cupy_array(
    ucxx_loop,
):
    cupy = pytest.importorskip("cupy")
    da = pytest.importorskip("dask.array")

    async with LocalCluster(
        protocol="ucxx", n_workers=1, threads_per_worker=1, asynchronous=True
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            assert cluster.scheduler_address.startswith("ucxx://")
            a = cupy.arange(10000)
            x = da.from_array(a, chunks=(10000,))
            b = await client.compute(x)
            cupy.testing.assert_array_equal(a, b)
