# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio

import pytest

import ucxx
from ucxx._lib_async.utils_test import wait_listener_client_handlers
from ucxx.types import Tag


@pytest.mark.asyncio
@pytest.mark.parametrize("transfer_api", ["am", "tag"])
async def test_message_probe(transfer_api):
    msg = bytearray(b"0" * 10)

    async def server_node(ep):
        # Wait for remote endpoint to close before probing the endpoint for
        # in-transit message and receiving it.
        while not ep.closed:
            await asyncio.sleep(0)  # Yield task

        if transfer_api == "am":
            assert ep._ep.am_probe() is True
            received = bytes(await ep.am_recv())
        else:
            assert ep._ctx.worker.tag_probe(Tag(ep._tags["msg_recv"])) is True
            received = bytearray(10)
            await ep.recv(received)
        assert received == msg

        await ep.close()
        listener.close()

    async def client_node(port):
        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            port,
        )
        if transfer_api == "am":
            await ep.am_send(msg)
        else:
            await ep.send(msg)
        await ep.close()

    listener = ucxx.create_listener(
        server_node,
    )
    await client_node(listener.port)

    wait_listener_client_handlers(listener)
    while not listener.closed:
        await asyncio.sleep(0.01)
