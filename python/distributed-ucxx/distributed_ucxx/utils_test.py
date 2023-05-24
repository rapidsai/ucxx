from __future__ import annotations

import asyncio
import logging
import sys

import pytest

from distributed.utils_test import cleanup, loop, loop_in_thread  # noqa: F401

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


# Let's make sure that UCX gets time to cancel
# progress tasks before closing the event loop.
@pytest.fixture(scope="function")
def ucxx_loop():
    """Allows UCX to cancel progress tasks before closing event loop.

    When UCX tasks are not completed in time (e.g., by unexpected Endpoint
    closure), clean up tasks before closing the event loop to prevent unwanted
    errors from being raised.
    """
    ucxx = pytest.importorskip("ucxx")

    event_loop = asyncio.new_event_loop()
    event_loop.set_exception_handler(ucxx_exception_handler)
    ucxx.reset()
    yield loop
    ucxx.reset()
    event_loop.close()

    # Reset also Distributed's UCX initialization, i.e., revert the effects of
    # `distributed.comm.ucx.init_once()`.
    import distributed_ucxx

    distributed_ucxx.ucxx = None
