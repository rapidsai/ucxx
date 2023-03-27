# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
from queue import Empty, Queue

import pytest

import ucxx


@pytest.mark.asyncio
@pytest.mark.parametrize("server_close_callback", [True, False])
async def test_close_callback(server_close_callback):
    closed = [False]

    def _close_callback():
        closed[0] = True

    async def server_node(ep):
        if server_close_callback is True:
            ep.set_close_callback(_close_callback)

    async def client_node(port):
        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            port,
        )
        if server_close_callback is False:
            ep.set_close_callback(_close_callback)

    listener = ucxx.create_listener(
        server_node,
    )
    await client_node(listener.port)
    while closed[0] is False:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("transfer_api", ["am", "tag", "tag_multi"])
async def test_cancel(transfer_api):
    if transfer_api == "am":
        pytest.skip("AM not implemented yet")

    q = Queue()

    async def server_node(ep):
        while True:
            try:
                # Make sure the listener doesn't return before the client schedules
                # the message to receive. If this is not done, UCXConnectionResetError
                # may be raised instead of UCXCanceledError.
                q.get(timeout=0.01)
                return
            except Empty:
                await asyncio.sleep(0)

    async def client_node(port):
        ep = await ucxx.create_endpoint(ucxx.get_address(), port)
        try:
            if transfer_api == "am":
                _, pending = await asyncio.wait(
                    [asyncio.create_task(ep.am_recv())], timeout=0.001
                )
            elif transfer_api == "tag":
                msg = bytearray(1)
                _, pending = await asyncio.wait(
                    [asyncio.create_task(ep.recv(msg))], timeout=0.001
                )
            else:
                _, pending = await asyncio.wait(
                    [asyncio.create_task(ep.recv_multi())], timeout=0.001
                )

            q.put("close")
            await asyncio.wait(pending)
            (pending,) = pending
            result = pending.result()
            assert isinstance(result, Exception)
            raise result
        except Exception as e:
            await ep.close()
            raise e

    listener = ucxx.create_listener(server_node)
    with pytest.raises(
        ucxx.exceptions.UCXCanceledError,
        # TODO: Add back custom UCXCanceledError messages?
    ):
        await client_node(listener.port)
    listener.close()
