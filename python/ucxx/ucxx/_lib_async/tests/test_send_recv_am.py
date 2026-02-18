# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
from functools import partial

import numpy as np
import pytest

import ucxx
from ucxx._lib.libucxx import PythonAmSendMemoryTypePolicy
from ucxx._lib_async.utils_test import wait_listener_client_handlers

msg_sizes = [0] + [2**i for i in range(0, 25, 4)]
iov_msg_sizes = [10] + [2**i for i in range(4, 25, 4)]


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


def simple_server(size, recv, memory_type_policy=None):
    async def server(ep):
        recv = await ep.am_recv()
        await ep.am_send(recv, memory_type_policy=memory_type_policy)
        await ep.close()

    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("recv_wait", [True, False])
@pytest.mark.parametrize("data", get_data())
@pytest.mark.parametrize(
    "memory_type_policy",
    [None, PythonAmSendMemoryTypePolicy.FallbackToHost],
)
async def test_send_recv_am(size, recv_wait, data, memory_type_policy):
    rndv_thresh = 8192
    ucxx.init(options={"RNDV_THRESH": str(rndv_thresh)})

    msg = data["generator"](size)

    recv = []
    listener = ucxx.create_listener(
        simple_server(size, recv, memory_type_policy=memory_type_policy)
    )
    num_clients = 1
    clients = [
        await ucxx.create_endpoint(ucxx.get_address(), listener.port)
        for i in range(num_clients)
    ]
    if recv_wait:
        # By sleeping here we ensure that the listener's
        # ep.am_recv call will have to wait, rather than return
        # immediately as receive data is already available.
        await asyncio.sleep(1)
    await asyncio.gather(
        *(c.am_send(msg, memory_type_policy=memory_type_policy) for c in clients)
    )
    recv_msgs = await asyncio.gather(*(c.am_recv() for c in clients))

    for recv_msg in recv_msgs:
        if data["memory_type"] == "cuda" and msg.nbytes < rndv_thresh:
            # Eager messages are always received on the host, if no custom host
            # allocator is registered, UCXX defaults to `np.array`.
            np.testing.assert_equal(recv_msg.view(np.int64), msg.get())
        else:
            data["validator"](recv_msg, msg)

    await asyncio.gather(*(c.close() for c in clients))
    await wait_listener_client_handlers(listener)


def simple_user_header_server(user_header_echo):
    async def server(ep):
        recv_buffer, recv_header = await ep.am_recv_with_header()
        # Echo back with the same user header the client sent
        await ep.am_send(recv_buffer, user_header=recv_header)
        user_header_echo.append(recv_header)
        await ep.close()

    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_am_user_header(size):
    ucxx.init()

    msg = bytearray(b"m" * size)
    user_header = b"test-header-\x00\x01\xff"

    server_received_headers = []
    listener = ucxx.create_listener(
        simple_user_header_server(server_received_headers)
    )
    ep = await ucxx.create_endpoint(ucxx.get_address(), listener.port)

    await ep.am_send(msg, user_header=user_header)
    recv_msg, recv_header = await ep.am_recv_with_header()

    assert bytes(recv_msg) == bytes(msg)
    assert recv_header == user_header

    # Verify the server also received the header correctly
    assert len(server_received_headers) == 1
    assert server_received_headers[0] == user_header

    await ep.close()
    await wait_listener_client_handlers(listener)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_am_empty_user_header(size):
    """Test that recv_header is empty bytes when no user header is sent."""
    ucxx.init()

    msg = bytearray(b"m" * size)

    listener = ucxx.create_listener(simple_server(size, [], memory_type_policy=None))
    ep = await ucxx.create_endpoint(ucxx.get_address(), listener.port)

    await ep.am_send(msg)
    recv_msg, recv_header = await ep.am_recv_with_header()

    assert bytes(recv_msg) == bytes(msg)
    assert recv_header == b""

    await ep.close()
    await wait_listener_client_handlers(listener)


def simple_iov_server():
    async def server(ep):
        recv = await ep.am_recv()
        await ep.am_send(recv)
        await ep.close()

    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("size", iov_msg_sizes)
async def test_send_recv_am_iov(size):
    ucxx.init()

    msg = bytearray(b"m" * size)
    mid = size // 2
    seg1 = msg[:mid]
    seg2 = msg[mid:]

    listener = ucxx.create_listener(simple_iov_server())
    ep = await ucxx.create_endpoint(ucxx.get_address(), listener.port)

    await ep.am_send_iov([seg1, seg2])
    recv_msg = await ep.am_recv()

    assert bytes(recv_msg) == bytes(msg)

    await ep.close()
    await wait_listener_client_handlers(listener)


def simple_iov_user_header_server(server_received_headers):
    async def server(ep):
        recv_buffer, recv_header = await ep.am_recv_with_header()
        server_received_headers.append(recv_header)
        await ep.am_send(recv_buffer)
        await ep.close()

    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("size", iov_msg_sizes)
async def test_send_recv_am_iov_user_header(size):
    ucxx.init()

    msg = bytearray(b"m" * size)
    mid = size // 2
    seg1 = msg[:mid]
    seg2 = msg[mid:]
    user_header = b"iov-header-data"

    server_received_headers = []
    listener = ucxx.create_listener(
        simple_iov_user_header_server(server_received_headers)
    )
    ep = await ucxx.create_endpoint(ucxx.get_address(), listener.port)

    await ep.am_send_iov([seg1, seg2], user_header=user_header)
    recv_msg, recv_header = await ep.am_recv_with_header()

    assert bytes(recv_msg) == bytes(msg)
    assert recv_header == user_header
    assert len(server_received_headers) == 1
    assert server_received_headers[0] == user_header

    await ep.close()
    await wait_listener_client_handlers(listener)
