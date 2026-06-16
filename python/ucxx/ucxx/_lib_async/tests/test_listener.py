# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import inspect
import logging
import weakref

import pytest

from ucxx._lib_async import listener as listener_mod


class _ActiveClients:
    def __init__(self, on_dec=None):
        self._on_dec = on_dec

    def inc(self, ident):
        pass

    def dec(self, ident):
        if self._on_dec is not None:
            self._on_dec()
        pass


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


class _MessageTruncatedError(Exception):
    pass


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
            ctx=ctx,
            func=callback,
            endpoint_error_handling=True,
            connect_timeout=1,
            ident=1,
            active_clients=_ActiveClients(),
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
async def test_listener_handler_clears_context_before_final_decrement(monkeypatch):
    async def fail_peer_info_exchange(*args, **kwargs):
        raise _MessageTruncatedError("message truncated")

    def check_listener_frame():
        frame = inspect.currentframe()
        assert frame is not None
        listener_frame = frame.f_back.f_back
        assert listener_frame.f_code.co_name == "_listener_handler_coroutine"

        listener_locals = listener_frame.f_locals
        assert listener_locals["ctx"] is None
        assert listener_locals["endpoint"] is None
        assert listener_locals["conn_request"] is None
        assert listener_locals["ep"] is None

    monkeypatch.setattr(
        listener_mod,
        "UCXMessageTruncatedError",
        _MessageTruncatedError,
    )
    monkeypatch.setattr(
        listener_mod,
        "exchange_peer_info",
        fail_peer_info_exchange,
    )

    await listener_mod._listener_handler_coroutine(
        conn_request=_ConnectionRequest(),
        ctx=_Context(),
        func=lambda ep: None,
        endpoint_error_handling=True,
        connect_timeout=1,
        ident=1,
        active_clients=_ActiveClients(on_dec=check_listener_frame),
    )
