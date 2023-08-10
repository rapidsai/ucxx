import asyncio
from functools import partial

import numpy as np
import pytest
from utils import wait_listener_client_handlers

import ucxx

msg_sizes = [0] + [2**i for i in range(0, 25, 4)]


def _bytearray_assert_equal(a, b):
    assert a == b


def get_data():
    ret = [
        {
            "allocator": bytearray,
            "generator": lambda n: bytearray(b"m" * n),
            "validator": lambda recv, exp: _bytearray_assert_equal(bytes(recv), exp),
            "memory_type": "host",
        },
        {
            "allocator": partial(np.ones, dtype=np.uint8),
            "generator": partial(np.arange, dtype=np.int64),
            "validator": lambda recv, exp: np.testing.assert_equal(
                recv.view(np.int64), exp
            ),
            "memory_type": "host",
        },
    ]

    try:
        import cupy as cp

        ret.append(
            {
                "allocator": partial(cp.ones, dtype=np.uint8),
                "generator": partial(cp.arange, dtype=np.int64),
                "validator": lambda recv, exp: cp.testing.assert_array_equal(
                    cp.asarray(recv).view(np.int64), exp
                ),
                "memory_type": "cuda",
            }
        )
    except ImportError:
        pass

    return ret


def simple_server(size, recv):
    async def server(ep):
        recv.append(await ep.am_recv())
        await ep.close()

    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("recv_wait", [True, False])
@pytest.mark.parametrize("data", get_data())
async def test_send_recv_am(size, recv_wait, data):
    rndv_thresh = 8192
    ucxx.init(options={"RNDV_THRESH": str(rndv_thresh)})

    msg = data["generator"](size)

    recv = []
    listener = ucxx.create_listener(simple_server(size, recv))
    num_clients = 1
    clients = [
        await ucxx.create_endpoint(ucxx.get_address(), listener.port)
        for i in range(num_clients)
    ]
    for c in clients:
        if recv_wait:
            # By sleeping here we ensure that the listener's
            # ep.am_recv call will have to wait, rather than return
            # immediately as receive data is already available.
            await asyncio.sleep(1)
        await c.am_send(msg)

    while len(recv) == 0:
        await asyncio.sleep(0)

    if data["memory_type"] == "cuda" and msg.nbytes < rndv_thresh:
        # Eager messages are always received on the host, if no custom host
        # allocator is registered, UCXX defaults to `np.array`.
        np.testing.assert_equal(recv[0].view(np.int64), msg.get())
    else:
        data["validator"](recv[0], msg)

    await asyncio.gather(*(c.close() for c in clients))
    await wait_listener_client_handlers(listener)
