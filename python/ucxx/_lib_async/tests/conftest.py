# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import gc
import os

import pytest

import ucxx

# Prevent calls such as `cudf = pytest.importorskip("cudf")` from initializing
# a CUDA context. Such calls may cause tests that must initialize the CUDA
# context on the appropriate device to fail.
# For example, without `RAPIDS_NO_INITIALIZE=True`, `test_benchmark_cluster`
# will succeed if running alone, but fails when all tests are run in batch.
os.environ["RAPIDS_NO_INITIALIZE"] = "True"


@pytest.hookimpl()
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


# Let's make sure that UCX gets time to cancel
# progress tasks before closing the event loop.
@pytest.fixture()
def event_loop(scope="session"):
    loop = asyncio.new_event_loop()
    try:
        loop.set_exception_handler(handle_exception)
        ucxx.reset()
        yield loop
        ucxx.reset()
        loop.run_until_complete(asyncio.sleep(0))
    finally:
        loop.close()


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
    """
    timeout_marker = pyfuncitem.get_closest_marker("asyncio_timeout")
    slow_marker = pyfuncitem.get_closest_marker("slow")
    rerun_marker = pyfuncitem.get_closest_marker("rerun_on_failure")
    default_timeout = 600.0 if slow_marker else 60.0
    timeout = float(timeout_marker.args[0]) if timeout_marker else default_timeout
    if timeout <= 0.0:
        raise ValueError("The `pytest.mark.asyncio_timeout` value must be positive.")

    if rerun_marker and len(rerun_marker.args) >= 0:
        reruns = rerun_marker.args[0]
        if not isinstance(reruns, int) or reruns < 0:
            raise ValueError("The `pytest.mark.rerun` value must be a positive integer")
    else:
        reruns = 1

    if asyncio.iscoroutinefunction(pyfuncitem.obj) and timeout > 0.0:

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
