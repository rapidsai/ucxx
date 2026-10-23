# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import os
import pickle
import time

import numpy as np
import pytest

import ucxx
from ucxx._lib_async.utils_test import wait_listener_client_handlers

cudf = pytest.importorskip("cudf")
distributed = pytest.importorskip("distributed")
cuda = pytest.importorskip("numba.cuda")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "g",
    [
        lambda cudf: cudf.Series([1, 2, 3]),
        lambda cudf: cudf.Series([1, 2, 3], index=[4, 5, 6]),
        lambda cudf: cudf.Series([1, None, 3]),
        lambda cudf: cudf.Series(range(2**13)),
        lambda cudf: cudf.DataFrame({"a": np.random.random(1200000)}),
        lambda cudf: cudf.DataFrame({"a": range(2**20)}),
        lambda cudf: cudf.DataFrame({"a": range(2**26)}),
        lambda cudf: cudf.Series(),
        lambda cudf: cudf.DataFrame(),
        lambda cudf: cudf.DataFrame({"a": [], "b": []}),
        lambda cudf: cudf.DataFrame({"a": [1.0], "b": [2.0]}),
        lambda cudf: cudf.DataFrame(
            {"a": ["a", "b", "c", "d"], "b": ["a", "b", "c", "d"]}
        ),
        lambda cudf: cudf.datasets.timeseries(),  # ts index with ints, cats, floats
    ],
)
async def test_send_recv_cudf(g, request):
    from distributed.utils import nbytes

    def trace(role, phase, **fields):
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        print(
            f"UCXX_CUDF_TRACE time_ns={time.monotonic_ns()} pid={os.getpid()} "
            f"test={request.node.name} role={role} phase={phase} {details}",
            flush=True,
        )

    class UCX:
        def __init__(self, ep, role):
            self.ep = ep
            self.role = role

        async def write(self, cdf):
            trace(self.role, "serialize-start")
            header, _frames = cdf.serialize()
            frames = [pickle.dumps(header)] + _frames
            trace(
                self.role,
                "serialize-done",
                sizes=tuple(nbytes(frame) for frame in frames),
                cuda=tuple(
                    hasattr(frame, "__cuda_array_interface__") for frame in frames
                ),
            )

            # Send meta data
            trace(self.role, "send-count-start")
            await self.ep.send(np.array([len(frames)], dtype=np.uint64))
            trace(self.role, "send-count-done")
            trace(self.role, "send-memory-types-start")
            await self.ep.send(
                np.array(
                    [hasattr(f, "__cuda_array_interface__") for f in frames],
                    dtype=bool,
                )
            )
            trace(self.role, "send-memory-types-done")
            trace(self.role, "send-sizes-start")
            await self.ep.send(np.array([nbytes(f) for f in frames], dtype=np.uint64))
            trace(self.role, "send-sizes-done")
            # Send frames
            for index, frame in enumerate(frames):
                size = nbytes(frame)
                if size > 0:
                    trace(self.role, "send-frame-start", index=index, size=size)
                    await self.ep.send(frame)
                    trace(self.role, "send-frame-done", index=index)
            trace(self.role, "write-done")

        async def read(self):
            try:
                # Recv meta data
                nframes = np.empty(1, dtype=np.uint64)
                trace(self.role, "recv-count-start")
                await self.ep.recv(nframes)
                trace(self.role, "recv-count-done", count=int(nframes[0]))
                is_cudas = np.empty(nframes[0], dtype=bool)
                trace(self.role, "recv-memory-types-start")
                await self.ep.recv(is_cudas)
                trace(self.role, "recv-memory-types-done", cuda=tuple(is_cudas))
                sizes = np.empty(nframes[0], dtype=np.uint64)
                trace(self.role, "recv-sizes-start")
                await self.ep.recv(sizes)
                trace(self.role, "recv-sizes-done", sizes=tuple(sizes))
            except (
                ucxx.exceptions.UCXCanceledError,
                ucxx.exceptions.UCXCloseError,
            ) as e:
                msg = "SOMETHING TERRIBLE HAS HAPPENED IN THE TEST"
                raise e(msg)
            else:
                # Recv frames
                frames = []
                for index, (is_cuda, size) in enumerate(
                    zip(is_cudas.tolist(), sizes.tolist())
                ):
                    if size > 0:
                        if is_cuda:
                            frame = cuda.device_array((size,), dtype=np.uint8)
                        else:
                            frame = np.empty(size, dtype=np.uint8)
                        trace(self.role, "recv-frame-start", index=index, size=size)
                        await self.ep.recv(frame)
                        trace(self.role, "recv-frame-done", index=index)
                        frames.append(frame)
                    else:
                        if is_cuda:
                            frames.append(cuda.device_array((0,), dtype=np.uint8))
                        else:
                            frames.append(b"")
                trace(self.role, "read-done")
                return frames

    class UCXListener:
        def __init__(self):
            self.comm = None

        def start(self):
            async def serve_forever(ep):
                ucx = UCX(ep, "server")
                self.comm = ucx

            self.ucxx_server = ucxx.create_listener(serve_forever)

    uu = UCXListener()
    uu.start()
    trace("test", "listener-created", port=uu.ucxx_server.port)
    uu.address = ucxx.get_address()
    trace("test", "client-connect-start")
    uu.client = await ucxx.create_endpoint(uu.address, uu.ucxx_server.port)
    trace("test", "client-connect-done")
    ucx = UCX(uu.client, "client")
    await asyncio.sleep(0.2)
    trace("test", "payload-create-start")
    msg = g(cudf)
    trace("test", "payload-create-done")
    trace("test", "transfer-start")
    frames, _ = await asyncio.gather(uu.comm.read(), ucx.write(msg))
    trace("test", "transfer-done")
    trace("test", "deserialize-start")
    ucx_header = pickle.loads(frames[0])
    cudf_buffer = frames[1:]
    typ = type(msg)
    res = typ.deserialize(ucx_header, cudf_buffer)
    trace("test", "deserialize-done")

    from cudf.testing import assert_eq

    trace("test", "compare-start")
    assert_eq(res, msg)
    trace("test", "compare-done")
    trace("test", "server-close-start")
    await uu.comm.ep.close()
    trace("test", "server-close-done")
    trace("test", "client-close-start")
    await uu.client.close()
    trace("test", "client-close-done")

    assert uu.client.closed
    assert uu.comm.ep.closed
    trace("test", "listener-handlers-wait-start")
    await wait_listener_client_handlers(uu.ucxx_server)
    trace("test", "listener-handlers-wait-done")
