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
        f"resources={resource_ids} "
        f"notifier={notifier_state} "
        f"notifier_threads={notifier_threads}"
    )
    if include_objects:
        live_objects = [
            (type(obj).__name__, hex(id(obj)), hex(id(getattr(obj, "_ctx", None))))
            for obj in gc.get_objects()
            if type(obj).__module__
            in ("ucxx._lib_async.endpoint", "ucxx._lib_async.listener")
            and type(obj).__name__ in ("Endpoint", "Listener")
        ]
        details += (
            f" live_objects={live_objects[:20]} live_object_count={len(live_objects)}"
        )
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
