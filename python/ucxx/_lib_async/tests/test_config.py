# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os
from unittest.mock import patch

import pytest
from ucxx._lib_async.utils_test import captured_logger

import ucxx


def test_get_config():
    with patch.dict(os.environ):
        # Unset to test default value
        if os.environ.get("UCX_TLS") is not None:
            del os.environ["UCX_TLS"]
        ucxx.reset()
        config = ucxx.get_config()
        assert isinstance(config, dict)
        assert config["TLS"] == "all"


@patch.dict(os.environ, {"UCX_SEG_SIZE": "2M"})
def test_set_env():
    ucxx.reset()
    config = ucxx.get_config()
    assert config["SEG_SIZE"] == os.environ["UCX_SEG_SIZE"]


@patch.dict(os.environ, {"UCX_SEG_SIZE": "2M"})
def test_init_options():
    ucxx.reset()
    options = {"SEG_SIZE": "3M"}
    # environment specification should be ignored
    ucxx.init(options)
    config = ucxx.get_config()
    assert config["SEG_SIZE"] == options["SEG_SIZE"]


@patch.dict(os.environ, {"UCX_SEG_SIZE": "4M"})
def test_init_options_and_env():
    ucxx.reset()
    options = {"SEG_SIZE": "3M"}  # Should be ignored
    ucxx.init(options, env_takes_precedence=True)
    config = ucxx.get_config()
    assert config["SEG_SIZE"] == os.environ["UCX_SEG_SIZE"]
    # Provided options dict was not modified.
    assert options == {"SEG_SIZE": "3M"}


@pytest.mark.skipif(
    ucxx.get_ucx_version() >= (1, 12, 0),
    reason="Beginning with UCX >= 1.12, it's only possible to validate "
    "UCP options but not options from other modules such as UCT. "
    "See https://github.com/openucx/ucx/issues/7519.",
)
def test_init_unknown_option():
    ucxx.reset()
    options = {"UNKNOWN_OPTION": "3M"}
    with pytest.raises(ucxx.exceptions.UCXInvalidParamError):
        ucxx.init(options)


def test_init_invalid_option():
    ucxx.reset()
    options = {"SEG_SIZE": "invalid-size"}
    with pytest.raises(ucxx.exceptions.UCXInvalidParamError):
        ucxx.init(options)


@patch.dict(os.environ, {"UCX_SEG_SIZE": "2M"})
def test_logging():
    """
    Test default logging configuration.
    """
    import logging

    root = logging.getLogger("ucx")

    # ucxx.init will only print INFO LINES
    with captured_logger(root, level=logging.INFO) as foreign_log:
        ucxx.reset()
        options = {"SEG_SIZE": "3M"}
        ucxx.init(options)
    assert len(foreign_log.getvalue()) > 0

    with captured_logger(root, level=logging.ERROR) as foreign_log:
        ucxx.reset()
        options = {"SEG_SIZE": "3M"}
        ucxx.init(options)

    assert len(foreign_log.getvalue()) == 0


@pytest.mark.parametrize(
    "progress_mode", ["blocking", "polling", "thread", "thread-polling"]
)
@pytest.mark.asyncio
async def test_python_future_warnings_env(progress_mode):
    with patch.dict(
        os.environ,
        {"UCXPY_PROGRESS_MODE": progress_mode, "UCXPY_ENABLE_PYTHON_FUTURE": "1"},
    ):
        ctx = (
            contextlib.nullcontext()
            if progress_mode.startswith("thread")
            else pytest.warns(
                UserWarning,
                match=f"Notifier thread requested, but {progress_mode} does "
                "not support it, using Python wait_yield().",
            )
        )
        with ctx:
            ucxx.init()


@pytest.mark.parametrize(
    "progress_mode", ["blocking", "polling", "thread", "thread-polling"]
)
@pytest.mark.asyncio
async def test_python_future_warnings_init_options(progress_mode):
    ctx = (
        contextlib.nullcontext()
        if progress_mode.startswith("thread")
        else pytest.warns(
            UserWarning,
            match=f"Notifier thread requested, but {progress_mode} does "
            "not support it, using Python wait_yield().",
        )
    )
    with ctx:
        ucxx.init(progress_mode=progress_mode, enable_python_future=True)


@pytest.mark.parametrize(
    "progress_mode", ["blocking", "polling", "thread", "thread-polling"]
)
@pytest.mark.asyncio
async def test_python_future_warnings_init_options_and_env(progress_mode):
    # Environment variables have higher priority. To test that we're getting the
    # proper warnings (or not getting them), we use reverse conditions here as to
    # what should trigger them with environment variables.
    kwargs = {
        "progress_mode": "blocking" if progress_mode.startswith("thread") else "thread",
        "enable_python_future": False,
    }

    with patch.dict(
        os.environ,
        {"UCXPY_PROGRESS_MODE": progress_mode, "UCXPY_ENABLE_PYTHON_FUTURE": "1"},
    ):
        ctx = (
            contextlib.nullcontext()
            if progress_mode.startswith("thread")
            else pytest.warns(
                UserWarning,
                match=f"Notifier thread requested, but {progress_mode} does "
                "not support it, using Python wait_yield().",
            )
        )
        with ctx:
            ucxx.init(**kwargs)
