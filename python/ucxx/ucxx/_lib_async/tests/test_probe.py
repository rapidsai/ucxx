# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio

import pytest

import ucxx
from ucxx._lib_async.utils_test import wait_listener_client_handlers
from ucxx.types import Tag


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_type", ["am", "tag", "tag_remove"])
async def test_message_probe(probe_type):
    msg = bytearray(b"0" * 10)

    async def server_node(ep):
        # Wait for remote endpoint to close before probing the endpoint for
        # in-transit message and receiving it.
        while not ep.closed:
            await asyncio.sleep(0)  # Yield task

        if probe_type == "am":
            assert ep._ep.am_probe() is True
            received = bytes(await ep.am_recv())
        elif probe_type == "tag":
            while not ep._ctx.worker.tag_probe(Tag(ep._tags["msg_recv"])).matched:
                ucxx.progress()
            received = bytearray(10)
            await ep.recv(received)
        elif probe_type == "tag_remove":
            while True:
                probe_info = ep._ctx.worker.tag_probe(
                    Tag(ep._tags["msg_recv"]), remove=True
                )
                if probe_info.matched:
                    break
                ucxx.progress()
            received = bytearray(10)
            await ep.recv_with_handle(received, probe_info.handle)  # type: ignore
        assert received == msg

        await ep.close()
        listener.close()

    async def client_node(port):
        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            port,
        )
        if probe_type == "am":
            await ep.am_send(msg)
        elif probe_type in ("tag", "tag_remove"):
            await ep.send(msg)
        await ep.close()

    listener = ucxx.create_listener(
        server_node,
    )
    await client_node(listener.port)

    wait_listener_client_handlers(listener)
    while not listener.closed:
        await asyncio.sleep(0.01)
