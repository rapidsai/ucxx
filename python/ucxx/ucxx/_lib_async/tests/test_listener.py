# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import concurrent.futures
import gc
import logging
import weakref

import pytest

import ucxx
import ucxx._lib.libucxx as ucx_api
from ucxx._lib_async import listener as listener_mod


class _ConnectionRequest:
    handle = 1

    def raise_on_error(self):
        raise RuntimeError("connection reset by remote peer")


class _Context:
    pass


class _Endpoint:
    def __init__(self, endpoint, ctx, tags):
        self.ctx = ctx
        self._tags = tags


class _LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "log_message", "exception_message"),
    [
        (
            "peer-info",
            "Unexpected error in listener handler coroutine",
            "RuntimeError: peer info exchange failed",
        ),
        (
            "callback",
            "Uncaught listener callback error",
            "RuntimeError: listener callback failed",
        ),
    ],
)
async def test_listener_handler_exception_log_does_not_retain_context(
    monkeypatch,
    failure,
    log_message,
    exception_message,
):
    async def fail_peer_info_exchange(*args, **kwargs):
        raise RuntimeError("peer info exchange failed")

    async def exchange_peer_info(*args, **kwargs):
        return {"msg_tag": 2}

    async def fail_callback(ep):
        raise RuntimeError("listener callback failed")

    def noop_callback(ep):
        pass

    if failure == "peer-info":
        exchange_peer_info = fail_peer_info_exchange
        callback = noop_callback
    else:
        callback = fail_callback

    monkeypatch.setattr(
        listener_mod,
        "exchange_peer_info",
        exchange_peer_info,
    )
    monkeypatch.setattr(
        listener_mod,
        "Endpoint",
        _Endpoint,
    )

    log_handler = _LogCaptureHandler()
    listener_mod.logger.addHandler(log_handler)
    try:
        ctx = _Context()
        ctx_ref = weakref.ref(ctx)

        await listener_mod._listener_handler_coroutine(
            conn_request=_ConnectionRequest(),
            ctx_ref=ctx_ref,
            func=callback,
            endpoint_error_handling=True,
            connect_timeout=1,
        )
        del ctx

        gc.collect()

        assert log_handler.records
        record = log_handler.records[-1]
        assert log_message in record.getMessage()
        assert record.exc_info is None
        assert exception_message in record.getMessage()
        assert ctx_ref() is None
    finally:
        listener_mod.logger.removeHandler(log_handler)


@pytest.mark.asyncio
async def test_listener_handler_tracker_waits_for_scheduled_future():
    tracker = listener_mod._ListenerHandlerTracker()
    gate = asyncio.Event()

    tracker.submit(gate.wait(), asyncio.get_running_loop())

    assert tracker.active_count == 1
    gate.set()
    await tracker.wait()
    assert tracker.active_count == 0


@pytest.mark.asyncio
async def test_listener_handler_tracker_retires_on_next_loop_turn(monkeypatch):
    future = concurrent.futures.Future()
    tracker = listener_mod._ListenerHandlerTracker()

    def run_coroutine_threadsafe(coroutine, event_loop):
        coroutine.close()
        return future

    monkeypatch.setattr(
        listener_mod.asyncio,
        "run_coroutine_threadsafe",
        run_coroutine_threadsafe,
    )

    tracker.submit(asyncio.sleep(0), asyncio.get_running_loop())
    future.set_result(None)

    # Completion alone is insufficient: run_coroutine_threadsafe may still own the
    # asyncio Task from the callback that completed this concurrent future.
    assert tracker.active_count == 1
    await asyncio.sleep(0)
    assert tracker.active_count == 0


def test_listener_handler_tracker_closes_coroutine_if_submission_fails(monkeypatch):
    tracker = listener_mod._ListenerHandlerTracker()
    coroutine = asyncio.sleep(0)

    def run_coroutine_threadsafe(coroutine, event_loop):
        raise RuntimeError("event loop is closed")

    monkeypatch.setattr(
        listener_mod.asyncio,
        "run_coroutine_threadsafe",
        run_coroutine_threadsafe,
    )

    with pytest.raises(RuntimeError, match="event loop is closed"):
        tracker.submit(coroutine, object())

    assert coroutine.cr_frame is None
    assert tracker.active_count == 0


@pytest.mark.asyncio
async def test_listener_handler_ignores_expired_context():
    callback_called = False

    def callback(ep):
        nonlocal callback_called
        callback_called = True

    ctx = _Context()
    ctx_ref = weakref.ref(ctx)
    del ctx
    gc.collect()

    await listener_mod._listener_handler_coroutine(
        conn_request=_ConnectionRequest(),
        ctx_ref=ctx_ref,
        func=callback,
        endpoint_error_handling=True,
        connect_timeout=1,
    )

    assert callback_called is False


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_close", [True, False])
async def test_listener_released_while_server_endpoint_alive(explicit_close):
    """Listener callback must not run after the listener is closed or released.

    Server endpoints keep the C++ listener alive, releasing the Python listener must
    still stop the UCP listener, otherwise new connection requests would execute the
    listener callback with its already released callback data.
    """
    server_endpoints = []

    async def handler(ep):
        server_endpoints.append(ep)

    log_handler = _LogCaptureHandler()
    ucx_logger = logging.getLogger("ucx")
    ucx_logger.addHandler(log_handler)
    client = None
    try:
        listener = ucxx.create_listener(handler)
        port = listener.port
        client = await ucxx.create_endpoint(ucxx.get_address(), port)
        while not server_endpoints:
            await asyncio.sleep(0.01)

        listener_ref = weakref.ref(listener._listener)
        if explicit_close:
            listener.close()
            assert listener.closed
        del listener
        gc.collect()
        assert listener_ref() is None

        # Reuse memory of the released callback data.
        junk = [{"x": i, "y": str(i)} for i in range(200000)]

        # Connect with a low-level endpoint: with the listener closed the connection
        # is rejected, the listener callback must not run.
        ctx = ucxx.core._get_ctx()
        ucx_ep = ucx_api.UCXEndpoint.create(ctx.worker, ucxx.get_address(), port, True)
        deadline = asyncio.get_running_loop().time() + 1.0
        while asyncio.get_running_loop().time() < deadline:
            if not ctx.progress_mode.startswith("thread"):
                ctx.worker.progress()
            await asyncio.sleep(0.01)
        del ucx_ep, ctx, junk

        assert len(server_endpoints) == 1
        errors = [
            r.getMessage()
            for r in log_handler.records
            if "listener callback" in r.getMessage()
        ]
        assert errors == []
    finally:
        ucx_logger.removeHandler(log_handler)
        if client is not None:
            await client.close()
        for ep in server_endpoints:
            await ep.close()
        server_endpoints.clear()
