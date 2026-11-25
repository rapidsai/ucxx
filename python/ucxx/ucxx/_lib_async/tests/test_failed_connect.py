# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import gc
import os
import weakref

import pytest

import ucxx
import ucxx._lib.libucxx as ucx_api
from ucxx._lib.libucxx import UCXCanceled

# Safety nets only, the tests synchronize on events and request states.
SAFETY_TIMEOUT = 30.0


class Sentinel:
    """Object whose lifetime is observed through a weak reference."""


def _init(enable_python_future):
    progress_mode = os.environ.get("UCXPY_PROGRESS_MODE", "thread")
    if enable_python_future and not progress_mode.startswith("thread"):
        pytest.skip(f"Python futures are not supported in '{progress_mode}' mode")
    ucxx.init(enable_python_future=enable_python_future)
    assert ucxx.core._ctx.worker.enable_python_future == enable_python_future


async def _wait_until(predicate, progress=None):
    while not predicate():
        if progress is not None:
            progress()
        await asyncio.sleep(0)


def _progress():
    ctx = ucxx.core._ctx
    return None if ctx.progress_mode.startswith("thread") else ctx.worker.progress


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_python_future", [False, True])
async def test_failed_request_releases_awaiting_frames(enable_python_future):
    """A failed request must not keep the frames that awaited it alive.

    With Python futures, the exception raised by ``UCXRequest.wait()`` used to be
    the one stored in the future owned by the C++ request, and its traceback
    references the awaiting frames, which reference the request: a reference cycle
    through C++ that the garbage collector cannot free.
    """
    _init(enable_python_future)

    server_eps = []
    listener = ucxx.create_listener(server_eps.append)
    ep = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
    await _wait_until(lambda: len(server_eps) == 1, _progress())

    async def recv(sentinel):
        await ep.recv(bytearray(8))

    sentinel = Sentinel()
    sentinel_ref = weakref.ref(sentinel)
    task = asyncio.create_task(recv(sentinel))
    del sentinel
    # The request is submitted in the same step that increments the counter.
    await _wait_until(lambda: ep._recv_count == 1)

    ep.abort()
    with pytest.raises(UCXCanceled):
        await asyncio.wait_for(task, timeout=SAFETY_TIMEOUT)
    del task
    # The event loop handle that resumed this coroutine references the task.
    await asyncio.sleep(0)
    gc.collect()

    assert sentinel_ref() is None

    server_eps[0].abort()
    listener.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_python_future", [False, True])
@pytest.mark.parametrize("peer", ["closes", "silent"])
async def test_failed_create_endpoint_releases_resources(
    monkeypatch, enable_python_future, peer
):
    """A failed ``create_endpoint()`` must close its endpoint and release the context.

    The peer is a raw listener that never exchanges peer information: it either
    closes the endpoint after accepting the connection, failing the exchange, or keeps
    it open without answering, so that the exchange times out.
    """
    _init(enable_python_future)
    loop = asyncio.get_running_loop()
    ctx_ref = weakref.ref(ucxx.core._ctx)

    accepted = asyncio.Event()
    server_eps = []

    def on_connection(server_ep):
        server_eps.append(server_ep)
        loop.call_soon_threadsafe(accepted.set)

    raw_listener = ucx_api.UCXListener.create(
        worker=ucxx.core._ctx.worker,
        port=0,
        cb_func=on_connection,
        deliver_endpoint=True,
    )

    client_eps = []
    create = ucx_api.UCXEndpoint.create

    def recording_create(*args, **kwargs):
        client_eps.append(create(*args, **kwargs))
        return client_eps[-1]

    monkeypatch.setattr(ucx_api.UCXEndpoint, "create", recording_create)

    connect_timeout = SAFETY_TIMEOUT if peer == "closes" else 0.1
    client = asyncio.create_task(
        ucxx.create_endpoint(
            ucxx.get_address(), raw_listener.port, connect_timeout=connect_timeout
        )
    )
    progress = _progress()
    while not accepted.is_set():
        if progress is not None:
            progress()
        await asyncio.sleep(0)
    if peer == "closes":
        server_eps[0].close_blocking(period=10**10, max_attempts=1)

    with pytest.raises(Exception) as excinfo:
        await asyncio.wait_for(client, timeout=2 * SAFETY_TIMEOUT)
    if peer == "silent":
        assert excinfo.type is asyncio.TimeoutError
    else:
        assert excinfo.type is not asyncio.TimeoutError
    del client, excinfo
    # The event loop handle that resumed this coroutine references the client task.
    await asyncio.sleep(0)

    # The endpoint is closed even though it is still referenced (here by
    # `client_eps`, usually by the traceback of the exception).
    assert len(client_eps) == 1
    assert not client_eps[0].alive

    monkeypatch.undo()
    client_eps.clear()
    for server_ep in server_eps:
        server_ep.close_blocking(period=10**10, max_attempts=1)
    server_eps.clear()
    raw_listener.close()
    del raw_listener

    ucxx.reset()
    assert ctx_ref() is None

