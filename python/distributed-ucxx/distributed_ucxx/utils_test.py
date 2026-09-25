# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager

import pytest

from distributed.utils_test import (  # noqa: F401
    check_thread_leak,
    cleanup,
    gen_test as distributed_gen_test,
    loop,
    loop_in_thread,
)

import ucxx

try:
    from pytest_timeout import is_debugging
except ImportError:

    def is_debugging() -> bool:
        # The pytest_timeout logic is more sophisticated. Not only debuggers
        # attach a trace callback but vendoring the entire logic is not worth it
        return sys.gettrace() is not None


logger = logging.getLogger(__name__)


def _direct_referrer_snapshot(obj):
    owners = []
    for referrer in gc.get_referrers(obj):
        referrer_type = type(referrer)
        owner = {
            "type": f"{referrer_type.__module__}.{referrer_type.__name__}",
            "id": hex(id(referrer)),
        }
        if isinstance(referrer, dict):
            keys = [repr(key)[:100] for key, value in referrer.items() if value is obj]
            # Exclude this diagnostic function's own locals mapping.
            if keys == ["'obj'"]:
                continue
            owner["keys"] = keys[:10]
        elif isinstance(referrer, (list, tuple)):
            owner["indices"] = [
                index for index, value in enumerate(referrer) if value is obj
            ][:10]
        elif isinstance(referrer, set):
            owner["contains_object"] = obj in referrer
        owners.append(owner)
    return owners[:20]


logging_levels = {
    name: logger.level
    for name, logger in logging.root.manager.loggerDict.items()
    if isinstance(logger, logging.Logger)
}


def ucxx_exception_handler(event_loop, context):
    """UCX exception handler for `ucxx_loop` during test.

    Prints the exception and its message.

    Parameters
    ----------
    loop: object
        Reference to the running event loop
    context: dict
        Dictionary containing exception details.
    """
    msg = context.get("exception", context["message"])
    print(msg)


def _nanny_lifecycle_snapshot(phase, include_objects=False):
    ctx = ucxx.core._ctx
    notifier = None if ctx is None else ctx.notifier_thread
    resources = None if ctx is None else getattr(ctx, "_dask_resources", None)
    try:
        resource_ids = None if resources is None else tuple(sorted(resources))
    except RuntimeError:
        resource_ids = "mutating"
    notifier_threads = tuple(
        (thread.ident, thread.is_alive())
        for thread in threading.enumerate()
        if thread.name == "UCX-Py Async Notifier Thread"
    )
    notifier_state = None if notifier is None else (notifier.ident, notifier.is_alive())
    details = (
        f"UCXX_NANNY_TRACE time_ns={time.monotonic_ns()} pid={os.getpid()} "
        f"phase={phase} ctx={None if ctx is None else hex(id(ctx))} "
        f"progress_mode={None if ctx is None else ctx.progress_mode} "
        f"delayed_submission={None if ctx is None else ctx.enable_delayed_submission} "
        f"python_future={None if ctx is None else ctx.enable_python_future} "
        f"resources={resource_ids} "
        f"notifier={notifier_state} "
        f"notifier_threads={notifier_threads}"
    )
    if include_objects:
        live_objects = []
        live_listeners = []
        live_comms = []
        for obj in gc.get_objects():
            obj_type = type(obj)
            if (
                obj_type.__module__ == "ucxx._lib_async.endpoint"
                and obj_type.__name__ == "Endpoint"
            ):
                endpoint = getattr(obj, "_ep", None)
                context = getattr(obj, "_ctx", None)
                live_objects.append(
                    {
                        "type": "Endpoint",
                        "id": hex(id(obj)),
                        "ctx": None if context is None else hex(id(context)),
                        "ucp_ep": None if endpoint is None else hex(id(endpoint)),
                        "send_count": getattr(obj, "_send_count", None),
                        "recv_count": getattr(obj, "_recv_count", None),
                        "finished_recv_count": getattr(
                            obj, "_finished_recv_count", None
                        ),
                        "shutting_down_peer": getattr(obj, "_shutting_down_peer", None),
                    }
                )
            elif (
                obj_type.__module__ == "ucxx._lib_async.listener"
                and obj_type.__name__ == "Listener"
            ):
                listener = getattr(obj, "_listener", None)
                context = getattr(obj, "_ctx", None)
                tracker = getattr(obj, "_handler_tracker", None)
                live_listeners.append(
                    {
                        "id": hex(id(obj)),
                        "ctx": None if context is None else hex(id(context)),
                        "ucp_listener": None if listener is None else hex(id(listener)),
                        "active_clients": getattr(tracker, "active_count", None),
                    }
                )
            elif (
                obj_type.__module__ == "distributed_ucxx.ucxx"
                and obj_type.__name__ == "UCXX"
            ):
                endpoint = getattr(obj, "_ep", None)
                live_comms.append(
                    {
                        "id": hex(id(obj)),
                        "endpoint": None if endpoint is None else hex(id(endpoint)),
                        "endpoint_handle": (
                            None
                            if getattr(obj, "_ep_handle", None) is None
                            else hex(obj._ep_handle)
                        ),
                        "resource_id": getattr(obj, "_resource_id", None),
                        "closed": getattr(obj, "_closed", None),
                        "has_close_callback": getattr(obj, "_has_close_callback", None),
                        "local_addr": getattr(obj, "_local_addr", None),
                        "peer_addr": getattr(obj, "_peer_addr", None),
                        "direct_referrers": _direct_referrer_snapshot(obj),
                    }
                )
        details += (
            f" endpoints={live_objects[:20]} endpoint_count={len(live_objects)}"
            f" listeners={live_listeners[:20]} listener_count={len(live_listeners)}"
            f" comms={live_comms[:20]} comm_count={len(live_comms)}"
        )
        frames = sys._current_frames()
        notifier_stacks = {
            thread.name: "".join(traceback.format_stack(frames[thread.ident]))
            for thread in threading.enumerate()
            if thread.name == "UCX-Py Async Notifier Thread" and thread.ident in frames
        }
        details += f" notifier_stacks={notifier_stacks}"
    return details


@contextmanager
def _nanny_lifecycle_diagnostics(request):
    if request.node.name != "test_nanny_closed_by_keyboard_interrupt":
        yield lambda phase: None
        return

    snapshots = [_nanny_lifecycle_snapshot("ready")]
    try:
        yield lambda phase: snapshots.append(_nanny_lifecycle_snapshot(phase))
    except BaseException:
        snapshots.append(_nanny_lifecycle_snapshot("failure", include_objects=True))
        print("\n".join(snapshots), flush=True)
        ucxx_module = sys.modules.get("distributed_ucxx.ucxx")
        trace = (
            None if ucxx_module is None else getattr(ucxx_module, "frame_trace", None)
        )
        if trace is not None:
            trace.dump()
        raise


# Let's make sure that UCX gets time to cancel
# progress tasks before closing the event loop.
@pytest.fixture(scope="function")
def ucxx_loop(request):
    """Allows UCX to cancel progress tasks before closing event loop.

    When UCX tasks are not completed in time (e.g., by unexpected Endpoint
    closure), clean up tasks before closing the event loop to prevent unwanted
    errors from being raised.

    Additionally add an `ignore_alive_references` marker that will override
    checks for alive references to `ApplicationContext`. Use sparingly!
    """
    marker = request.node.get_closest_marker("ignore_alive_references")
    ignore_alive_references = False if marker is None else marker.args[0]

    event_loop = asyncio.new_event_loop()
    event_loop.set_exception_handler(ucxx_exception_handler)

    # Create and reset context before running. The first test that runs during the
    # `pytest` process lifetime creates a `_DummyThread` instance which violates
    # thread checking from `distributed.utils_test.check_thread_leak()`, if we
    # instantiate a and reset a context before `yield loop`, that doesn't fail
    # during the `check_thread_leak()` check below.
    ucxx.core._get_ctx()
    ucxx.reset()

    with _nanny_lifecycle_diagnostics(request) as snapshot, check_thread_leak():
        yield loop
        snapshot("before-reset")
        if ignore_alive_references:
            try:
                ucxx.reset()
            except ucxx.exceptions.UCXError as e:
                if (
                    len(e.args) > 0
                    and "The following objects are still referencing ApplicationContext"
                    in e.args[0]
                ):
                    print(
                        "ApplicationContext still has alive references but this test "
                        f"is ignoring them. Original error:\n{e}",
                        flush=True,
                    )
                else:
                    raise e
        else:
            ucxx.reset()
        snapshot("after-reset")
        event_loop.close()

        # Reset also Distributed's UCX initialization, i.e., revert the effects of
        # `distributed.comm.ucx.init_once()`.
        import distributed_ucxx

        distributed_ucxx.ucxx = None


def gen_test(**kwargs):
    assert "clean_kwargs" not in kwargs
    return distributed_gen_test(clean_kwargs={"threads": False}, **kwargs)
