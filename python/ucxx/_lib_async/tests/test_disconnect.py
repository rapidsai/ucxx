# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import logging
import multiprocessing as mp
from io import StringIO
from queue import Empty

import numpy as np
import pytest
from ucxx._lib_async.utils import get_event_loop
from ucxx._lib_async.utils_test import wait_listener_client_handlers

import ucxx

mp = mp.get_context("spawn")


async def mp_queue_get_nowait(queue):
    while True:
        try:
            return queue.get_nowait()
        except Empty:
            pass
        await asyncio.sleep(0.01)


def _test_shutdown_unexpected_closed_peer_server(
    client_queue, server_queue, endpoint_error_handling
):
    global ep_alive
    ep_alive = None

    async def run():
        async def server_node(ep):
            try:
                global ep_alive

                await ep.send(np.arange(100, dtype=np.int64))
                # Waiting for signal to close the endpoint
                await mp_queue_get_nowait(server_queue)

                # At this point, the client should have died and the endpoint
                # is not alive anymore. `True` only when endpoint error
                # handling is enabled.
                ep_alive = ep.alive

                await ep.close()
            finally:
                listener.close()

        listener = ucxx.create_listener(
            server_node, endpoint_error_handling=endpoint_error_handling
        )
        client_queue.put(listener.port)
        await wait_listener_client_handlers(listener)
        while not listener.closed:
            await asyncio.sleep(0.1)

    log_stream = StringIO()
    logging.basicConfig(stream=log_stream, level=logging.DEBUG)
    get_event_loop().run_until_complete(run())
    log = log_stream.getvalue()

    if endpoint_error_handling is True:
        assert ep_alive is False
    else:
        assert ep_alive
        assert log.find("""UCXError('<[Send shutdown]""") != -1

    ucxx.stop_notifier_thread()


def _test_shutdown_unexpected_closed_peer_client(
    client_queue, server_queue, endpoint_error_handling
):
    async def run():
        server_port = client_queue.get()
        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            server_port,
            endpoint_error_handling=endpoint_error_handling,
        )
        msg = np.empty(100, dtype=np.int64)
        await ep.recv(msg)

    get_event_loop().run_until_complete(run())

    ucxx.stop_notifier_thread()


@pytest.mark.parametrize("endpoint_error_handling", [True, False])
def test_shutdown_unexpected_closed_peer(caplog, endpoint_error_handling):
    """
    Test clean server shutdown after unexpected peer close

    This will causes some UCX warnings to be issued, but this as expected.
    The main goal is to assert that the processes exit without errors
    despite a somewhat messy initial state.
    """
    if endpoint_error_handling is False:
        pytest.xfail(
            "Temporarily xfailing, due to https://github.com/rapidsai/ucxx/issues/21"
        )
    if endpoint_error_handling is False and any(
        [
            t.startswith(i)
            for i in ("rc", "dc", "ud")
            for t in ucxx.get_active_transports()
        ]
    ):
        pytest.skip(
            "Endpoint error handling is required when rc, dc or ud"
            "transport is enabled"
        )

    client_queue = mp.Queue()
    server_queue = mp.Queue()
    p1 = mp.Process(
        target=_test_shutdown_unexpected_closed_peer_server,
        args=(client_queue, server_queue, endpoint_error_handling),
    )
    p1.start()
    p2 = mp.Process(
        target=_test_shutdown_unexpected_closed_peer_client,
        args=(client_queue, server_queue, endpoint_error_handling),
    )
    p2.start()
    p2.join()
    server_queue.put("client is down")
    p1.join()

    assert not p1.exitcode
    assert not p2.exitcode
