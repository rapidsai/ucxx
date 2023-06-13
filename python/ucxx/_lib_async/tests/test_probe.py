# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio

import pytest

import ucxx as ucxx


@pytest.mark.asyncio
@pytest.mark.parametrize("transfer_api", ["am", "tag"])
@pytest.mark.xfail(reason="https://github.com/rapidsai/ucxx/issues/19")
async def test_message_probe(transfer_api):
    msg = bytearray(b"0" * 10)

    async def server_node(ep):
        # Wait for remote endpoint to close before probing the endpoint for
        # in-transit message and receiving it.
        while not ep.closed():
            await asyncio.sleep(0)  # Yield task

        if transfer_api == "am":
            assert ep._ep.am_probe() is True
            received = bytes(await ep.am_recv())
        else:
            assert ep._ctx.worker.tag_probe(ep._tags["msg_recv"]) is True
            received = bytearray(10)
            await ep.recv(received)
        assert received == msg

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

    listener = ucxx.create_listener(
        server_node,
    )
    await client_node(listener.port)

    while not listener.closed():
        await asyncio.sleep(0.01)
