# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
from functools import partial

import pytest

import ucxx
from ucxx._lib_async.utils_test import wait_listener_client_handlers
from ucxx.types import Tag

Message = bytearray(b"0" * 10)
Listener = None


async def _server_node(ep, listener=None, coroutine=None):
    global Listener

    # Wait for remote endpoint to close before probing the endpoint for
    # in-transit message and receiving it.
    while not ep.closed:
        await asyncio.sleep(0)  # Yield task

    received = await coroutine(ep)

    assert received == Message

    await ep.close()
    Listener.close()


async def _server_node_context_tag_coroutine(ep):
    ctx = ep._ctx
    while not ctx.tag_probe(Tag(ep._tags["msg_recv"])).matched:
        ucxx.progress()
    received = bytearray(10)
    await ctx.recv(received, Tag(ep._tags["msg_recv"]))
    return received


async def _server_node_endpoint_tag_coroutine(ep):
    while not ep.tag_probe().matched:
        ucxx.progress()
    received = bytearray(10)
    await ep.recv(received)
    return received


async def _server_node_context_tag_remove_coroutine(ep):
    ctx = ep._ctx
    while True:
        probe_info = ctx.tag_probe(Tag(ep._tags["msg_recv"]), remove=True)
        if probe_info.matched:
            break
        ucxx.progress()
    received = bytearray(10)
    await ctx.recv_with_handle(received, probe_info.handle)  # type: ignore
    return received


async def _server_node_endpoint_tag_remove_coroutine(ep):
    while True:
        probe_info = ep.tag_probe(remove=True)
        if probe_info.matched:
            break
        ucxx.progress()
    received = bytearray(10)
    await ep.recv_with_handle(received, probe_info.handle)  # type: ignore
    return received


async def _server_node_am_coroutine(ep):
    assert ep._ep.am_probe() is True
    return bytes(await ep.am_recv())


async def _client_node(probe_type, port):
    ep = await ucxx.create_endpoint(
        ucxx.get_address(),
        port,
    )
    if probe_type == "am":
        await ep.am_send(Message)
    elif probe_type in ("tag", "tag_remove"):
        await ep.send(Message)
    await ep.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("api_type", ["endpoint", "context"])
@pytest.mark.parametrize("probe_type", ["am", "tag", "tag_remove"])
async def test_message_probe(api_type, probe_type):
    global Listener

    if api_type == "context" and probe_type == "am":
        pytest.skip("There is no AM probe directly in the context")

    if probe_type == "am":
        server_node_partial = partial(_server_node, coroutine=_server_node_am_coroutine)
    elif probe_type == "tag":
        if api_type == "context":
            server_node_partial = partial(
                _server_node, coroutine=_server_node_context_tag_coroutine
            )
        elif api_type == "endpoint":
            server_node_partial = partial(
                _server_node, coroutine=_server_node_endpoint_tag_coroutine
            )
    elif probe_type == "tag_remove":
        if api_type == "context":
            server_node_partial = partial(
                _server_node, coroutine=_server_node_context_tag_remove_coroutine
            )
        elif api_type == "endpoint":
            server_node_partial = partial(
                _server_node, coroutine=_server_node_endpoint_tag_remove_coroutine
            )

    Listener = ucxx.create_listener(
        server_node_partial,
    )
    await _client_node(probe_type, Listener.port)

    wait_listener_client_handlers(Listener)
    while not Listener.closed:
        await asyncio.sleep(0.01)
