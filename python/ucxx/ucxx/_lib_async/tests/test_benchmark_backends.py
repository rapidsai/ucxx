# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import sys

import pytest

import ucxx
from ucxx.benchmarks.backends.ucxx_async import _recv_terminal_ack, _send_terminal_ack
from ucxx.benchmarks.send_recv import parse_args


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enable_am", "multi"), [(False, False), (False, True), (True, False)]
)
async def test_async_benchmark_terminal_ack_waits_for_client(enable_am, multi):
    """The server must not close before the client completed its final response."""

    server_received_ack = asyncio.Event()

    async def server_handler(ep):
        if enable_am:
            received = await ep.am_recv()
            await ep.am_send(received)
        elif multi:
            received = await ep.recv_multi()
            await ep.send_multi(received)
        else:
            received = bytearray(1)
            await ep.recv(received)
            await ep.send(received)

        await _recv_terminal_ack(ep, enable_am)
        server_received_ack.set()
        await ep.close()

    listener = ucxx.create_listener(server_handler)
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)

    if enable_am:
        await client.am_send(bytearray(1))
        await client.am_recv()
    elif multi:
        await client.send_multi([bytearray(1), bytearray(1)])
        await client.recv_multi()
    else:
        await client.send(bytearray(1))
        response = bytearray(1)
        await client.recv(response)

    assert not server_received_ack.is_set()
    await _send_terminal_ack(client, enable_am)
    await asyncio.wait_for(server_received_ack.wait(), timeout=5)


def test_am_benchmark_rejects_device_memory(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["send_recv.py", "--enable-am", "--object_type", "cupy"],
    )

    with pytest.raises(RuntimeError, match="supports only `--object_type=numpy`"):
        parse_args()


def test_am_benchmark_accepts_host_memory(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["send_recv.py", "--enable-am"])

    assert parse_args().enable_am
