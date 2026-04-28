# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import functools
import gc
import inspect
import os

import pytest

import ucxx
from ucxx._lib_async.pytest_stash_keys import ASYNCIO_PLUGIN_TIMEOUT_STASH_KEY

# Prevent calls such as `cudf = pytest.importorskip("cudf")` from initializing
# a CUDA context. Such calls may cause tests that must initialize the CUDA
# context on the appropriate device to fail.
# For example, without `RAPIDS_NO_INITIALIZE=True`, `test_benchmark_cluster`
# will succeed if running alone, but fails when all tests are run in batch.
os.environ["RAPIDS_NO_INITIALIZE"] = "True"


def pytest_runtest_teardown(item, nextitem):
    gc.collect()


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    print(msg)


class CustomEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """Custom event loop policy providing custom event loop with UCXX setup/teardown."""

    def new_event_loop(self):
        loop = super().new_event_loop()
        loop.set_exception_handler(handle_exception)
        return loop


@pytest.fixture(scope="session")
def event_loop_policy():
    """Provide a custom event loop policy for the entire test session."""
    policy = CustomEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)
    return policy


@pytest.fixture(autouse=True)
def ucxx_setup_teardown():
    """Automatically setup and teardown UCX for each test."""
    ucxx.reset()
    yield
    ucxx.reset()
    # Let's make sure that UCX gets time to cancel
    # progress tasks before closing the event loop.
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Python 3.14+ raises if there is no event loop
        loop = None
    if loop is None:
        pass
    elif loop.is_running():
        # If loop is running, we can't run_until_complete
        # The cleanup will happen when the loop is closed
        pass
    else:
        try:
            loop.run_until_complete(asyncio.sleep(0))
        except RuntimeError:
            # Loop might already be closed
            pass


def _asyncio_plugin_timeout_seconds(item: pytest.Item) -> float:
    """Timeout used by asyncio test wrapper and exposed via ``config.stash``."""
    timeout_marker = item.get_closest_marker("asyncio_timeout")
    slow_marker = item.get_closest_marker("slow")
    default_timeout = 600.0 if slow_marker else 60.0
    return float(timeout_marker.args[0]) if timeout_marker else default_timeout


def pytest_runtest_setup(item: pytest.Item) -> None:
    """
    Record the effective asyncio timeout before the test body runs.

    ``compute_timeouts(pytestconfig)`` reads this from ``config.stash``. We use stash
    instead of ``config.cache`` because the cache backend is unreliable on
    ``pytest-xdist`` workers (values may not round-trip or may be isolated per hook).
    """
    if not isinstance(item, pytest.Function):
        return
    timeout = _asyncio_plugin_timeout_seconds(item)
    if timeout <= 0.0:
        raise ValueError("The `pytest.mark.asyncio_timeout` value must be positive.")
    item.config.stash[ASYNCIO_PLUGIN_TIMEOUT_STASH_KEY] = timeout


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """
    Inject the asyncio timeout BEFORE pytest-asyncio applies its MonkeyPatch.

    In pytest-asyncio 1.3.0 (asyncio_mode=auto) + Python 3.14, the framework
    replaces ``item.obj`` with a sync wrapper inside
    ``PytestAsyncioFunction.runtest()``, which is called during
    ``pytest_runtest_call``.  By the time ``pytest_pyfunc_call`` fires,
    ``item.obj`` is already a sync function and
    ``inspect.iscoroutinefunction()`` returns False.  Wrapping here instead
    guarantees ``asyncio.wait_for`` is present in the coroutine that
    pytest-asyncio eventually passes to its ``asyncio.Runner``.
    """
    if isinstance(item, pytest.Function) and inspect.iscoroutinefunction(item.obj):
        timeout = _asyncio_plugin_timeout_seconds(item)
        if timeout > 0.0:
            original = item.obj
            test_name = item.name

            @functools.wraps(original)
            async def timed_coroutine(*args, **kwargs):
                try:
                    return await asyncio.wait_for(
                        original(*args, **kwargs), timeout=timeout
                    )
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pytest.fail(f"{test_name} timed out after {timeout} seconds.")

            item.obj = timed_coroutine
    yield


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function):
    """
    Add timeout for tests, and optionally rerun on failure.

    Add timeout for tests with `pytest.mark.asyncio_timeout` marker as specified by the
    decorator, otherwise a default timeout of 60 seconds for regular tests and 600
    seconds for tests marked slow.

    Optionally rerun the test if it failed, for that the test has to be marked with
    `pytest.mark.rerun_on_failure(reruns)`. This is similar to `pytest-rerunfailures`,
    but that module closes the event loop before this function has awaited, making the
    two incompatible.

    The timeout value is made available to the test functions via `pytestconfig`. This
    can be used to determine internal timeouts, for example to ensure subprocesses
    timeout before the test timeout hits and thus prints internal information, such as
    the call stack. The timeout is stored in ``config.stash`` (see
    ``pytest_runtest_setup`` and ``compute_timeouts``).
    """
    timeout = _asyncio_plugin_timeout_seconds(pyfuncitem)
    pyfuncitem.config.stash[ASYNCIO_PLUGIN_TIMEOUT_STASH_KEY] = timeout
    rerun_marker = pyfuncitem.get_closest_marker("rerun_on_failure")

    if rerun_marker and len(rerun_marker.args) >= 0:
        reruns = rerun_marker.args[0]
        if not isinstance(reruns, int) or reruns < 0:
            raise ValueError("The `pytest.mark.rerun` value must be a positive integer")
    else:
        reruns = 1

    if inspect.iscoroutinefunction(pyfuncitem.obj) and timeout > 0.0:

        async def wrapped_obj(*args, **kwargs):
            for i in range(reruns):
                try:
                    try:
                        return await asyncio.wait_for(
                            inner_obj(*args, **kwargs), timeout=timeout
                        )
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pytest.fail(
                            f"{pyfuncitem.name} timed out after {timeout} seconds."
                        )
                except Exception as e:
                    if i == (reruns - 1):
                        raise e
                else:
                    break

        inner_obj = pyfuncitem.obj
        pyfuncitem.obj = wrapped_obj

    yield


def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers",
        "asyncio_timeout(timeout): cancels the test execution after the specified "
        "number of seconds",
    )
    config.addinivalue_line(
        "markers",
        "rerun_on_failure(reruns): reruns test if it fails for the specified number "
        "of reruns",
    )
    config.addinivalue_line("markers", "slow: mark test as slow to run")
