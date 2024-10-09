# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ucxx
from ucxx._lib_async.utils_test import wait_listener_client_handlers

np = pytest.importorskip("numpy")

msg_sizes = [2**i for i in range(0, 25, 4)]
# multi_sizes = [0, 1, 2, 3, 4, 8]
multi_sizes = [1, 2, 3, 4, 8]
dtypes = ["|u1", "<i8", "f8"]


def make_echo_server():
    """
    Returns an echo server that calls the function `create_empty_data(nbytes)`
    to create the data container.`
    """

    async def echo_server(ep):
        """
        Basic echo server for sized messages.
        We expect the other endpoint to follow the pattern::
        # size of the real message (in bytes)
        >>> await ep.send(msg_size)
        >>> await ep.send(msg)       # send the real message
        >>> await ep.recv(responds)  # receive the echo
        """
        msg = await ep.recv_multi()
        await ep.send_multi(msg)
        await ep.close()

    return echo_server


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("multi_size", multi_sizes)
async def test_send_recv_bytes(size, multi_size):
    send_msg = [bytearray(b"m" * size)] * multi_size

    listener = ucxx.create_listener(make_echo_server())
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
    await client.send_multi(send_msg)
    recv_msg = await client.recv_multi()
    for r, s in zip(recv_msg, send_msg):
        np.testing.assert_array_equal(r, s)
    await client.close()
    await wait_listener_client_handlers(listener)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("multi_size", multi_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_numpy(size, multi_size, dtype):
    send_msg = [np.arange(size, dtype=dtype)] * multi_size

    listener = ucxx.create_listener(make_echo_server())
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
    await client.send_multi(send_msg)
    recv_msg = await client.recv_multi()
    for r, s in zip(recv_msg, send_msg):
        np.testing.assert_array_equal(r.view(dtype), s)
    await wait_listener_client_handlers(listener)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("multi_size", multi_sizes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.flaky(reruns=3)
async def test_send_recv_cupy(size, multi_size, dtype):
    cupy = pytest.importorskip("cupy")

    send_msg = [cupy.arange(size, dtype=dtype)] * multi_size

    listener = ucxx.create_listener(make_echo_server())
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
    await client.send_multi(send_msg)
    recv_msg = await client.recv_multi()
    for r, s in zip(recv_msg, send_msg):
        cupy.testing.assert_array_equal(cupy.asarray(r).view(dtype), cupy.asarray(s))
    await wait_listener_client_handlers(listener)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("multi_size", multi_sizes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.flaky(reruns=3)
async def test_send_recv_numba(size, multi_size, dtype):
    cuda = pytest.importorskip("numba.cuda")

    ary = np.arange(size, dtype=dtype)
    send_msg = [cuda.to_device(ary)] * multi_size
    listener = ucxx.create_listener(make_echo_server())
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
    await client.send_multi(send_msg)
    recv_msg = await client.recv_multi()
    for r, s in zip(recv_msg, send_msg):
        np.testing.assert_array_equal(
            r.copy_to_host().view(dtype), s.copy_to_host().view(dtype)
        )
    await wait_listener_client_handlers(listener)
