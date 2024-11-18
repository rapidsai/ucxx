# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import multiprocessing as mp
import os
import re
from unittest.mock import patch

import numpy as np
import pytest

import ucxx
from ucxx._lib_async.utils import get_event_loop
from ucxx.testing import join_processes, terminate_process

mp = mp.get_context("spawn")


def _test_from_worker_address_error_server(q1, q2, error_type):
    async def run():
        address = bytearray(ucxx.get_worker_address())

        if error_type == "unreachable":
            # Shutdown worker, then send its address to client process via
            # multiprocessing.Queue
            ucxx.reset()
            q1.put(address)
        else:
            # Send worker address to client process via # multiprocessing.Queue,
            # wait for client to connect, then shutdown worker.
            q1.put(address)

            ep_ready = q2.get()
            assert ep_ready == "ready"

            ucxx.reset()

            # q1.put("disconnected")

    loop = get_event_loop()
    loop.run_until_complete(run())

    ucxx.stop_notifier_thread()

    loop.close()


def _test_from_worker_address_error_client(q1, q2, error_type):
    async def run():
        # Receive worker address from server via multiprocessing.Queue
        remote_address = ucxx.get_ucx_address_from_buffer(q1.get())
        if error_type == "unreachable":
            server_closed = q1.get()
            assert server_closed == "Server closed"

        if error_type == "unreachable":
            with pytest.raises(
                ucxx.exceptions.UCXError,
                match="Destination is unreachable|Endpoint timeout",
            ):
                # Here, two cases may happen:
                # 1. With TCP creating endpoint will immediately raise
                #    "Destination is unreachable"
                # 2. With rc/ud creating endpoint will succeed, but raise
                #    "Endpoint timeout" after UCX_UD_TIMEOUT seconds have passed.
                #    We need to keep progressing ucxx until timeout is raised.
                ep = await ucxx.create_endpoint_from_worker_address(remote_address)
                while ep.alive:
                    await asyncio.sleep(0)
                    if not ucxx.core._get_ctx().progress_mode.startswith("thread"):
                        ucxx.progress()
                ep._ep.raise_on_error()
        else:
            # Create endpoint to remote worker, and:
            #
            # 1. For timeout_am_send/timeout_send:
            #    - inform remote worker that local endpoint is ready for remote
            #      shutdown;
            #    - wait for remote worker to shutdown and confirm;
            #    - attempt to send message.
            #
            # 2. For timeout_am_recv/timeout_recv:
            #    - schedule ep.recv;
            #    - inform remote worker that local endpoint is ready for remote
            #      shutdown;
            #    - wait for it to shutdown and confirm
            #    - wait for recv message.
            ep = await ucxx.create_endpoint_from_worker_address(remote_address)

            if re.match("timeout.*send", error_type):
                q2.put("ready")

                # Wait for remote endpoint to disconnect
                while ep.alive:
                    await asyncio.sleep(0)
                    if not ucxx.core._get_ctx().progress_mode.startswith("thread"):
                        ucxx.progress()

                # TCP generally raises `UCXConnectionResetError`, whereas InfiniBand
                # raises `UCXEndpointTimeoutError`
                with pytest.raises(
                    (
                        ucxx.exceptions.UCXConnectionResetError,
                        ucxx.exceptions.UCXEndpointTimeoutError,
                    )
                ):
                    if error_type == "timeout_am_send":
                        await asyncio.wait_for(ep.am_send(np.zeros(10)), timeout=1.0)
                    else:
                        await asyncio.wait_for(
                            ep.send(np.zeros(10), tag=0, force_tag=True), timeout=1.0
                        )
            else:
                # TCP generally raises `UCXConnectionResetError`, whereas InfiniBand
                # raises `UCXEndpointTimeoutError`
                with pytest.raises(
                    (
                        ucxx.exceptions.UCXConnectionResetError,
                        ucxx.exceptions.UCXEndpointTimeoutError,
                    )
                ):
                    if error_type == "timeout_am_recv":
                        task = asyncio.wait_for(ep.am_recv(), timeout=3.0)
                    else:
                        msg = np.empty(10)
                        task = asyncio.wait_for(
                            ep.recv(msg, tag=0, force_tag=True), timeout=3.0
                        )

                    q2.put("ready")

                    while ep.alive:
                        await asyncio.sleep(0)
                        if not ucxx.core._get_ctx().progress_mode.startswith("thread"):
                            ucxx.progress()

                    await task

    loop = get_event_loop()
    loop.run_until_complete(run())

    ucxx.stop_notifier_thread()

    loop.close()


@pytest.mark.parametrize(
    "error_type",
    [
        "unreachable",
        "timeout_am_send",
        "timeout_am_recv",
        "timeout_send",
        "timeout_recv",
    ],
)
@patch.dict(
    os.environ,
    {
        "UCX_WARN_UNUSED_ENV_VARS": "n",
        # Set low timeouts to ensure tests quickly raise as expected
        "UCX_KEEPALIVE_INTERVAL": "100ms",
        "UCX_UD_TIMEOUT": "100ms",
    },
)
def test_from_worker_address_error(error_type):
    q1 = mp.Queue()
    q2 = mp.Queue()

    server = mp.Process(
        target=_test_from_worker_address_error_server,
        args=(q1, q2, error_type),
    )
    server.start()

    client = mp.Process(
        target=_test_from_worker_address_error_client,
        args=(q1, q2, error_type),
    )
    client.start()

    if error_type == "unreachable":
        server.join()
        q1.put("Server closed")

    join_processes([client, server], timeout=30)
    terminate_process(server)
    try:
        terminate_process(client)
    except RuntimeError as e:
        if ucxx.get_ucx_version() < (1, 12, 0):
            if all(t in error_type for t in ["timeout", "send"]):
                pytest.xfail(
                    "Requires https://github.com/openucx/ucx/pull/7527 with rc/ud."
                )
            elif all(t in error_type for t in ["timeout", "recv"]):
                pytest.xfail(
                    "Requires https://github.com/openucx/ucx/pull/7531 with rc/ud."
                )
        else:
            raise e
