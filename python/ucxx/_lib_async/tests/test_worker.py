# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import os
from unittest.mock import patch

import pytest

import ucxx


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_delayed_submission", [True, False])
@pytest.mark.parametrize("enable_python_future", [True, False])
async def test_worker_capabilities_args(
    enable_delayed_submission, enable_python_future
):
    ucxx.init(
        enable_delayed_submission=enable_delayed_submission,
        enable_python_future=enable_python_future,
    )

    worker = ucxx.core._get_ctx().worker

    assert worker.is_delayed_submission_enabled() is enable_delayed_submission
    assert worker.is_python_future_enabled() is enable_python_future


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_delayed_submission", [True, False])
@pytest.mark.parametrize("enable_python_future", [True, False])
async def test_worker_capabilities_env(enable_delayed_submission, enable_python_future):
    with patch.dict(
        os.environ,
        {
            "UCXPY_ENABLE_DELAYED_SUBMISSION": "1"
            if enable_delayed_submission
            else "0",
            "UCXPY_ENABLE_PYTHON_FUTURE": "1" if enable_python_future else "0",
        },
    ):
        ucxx.init()

        worker = ucxx.core._get_ctx().worker

        assert worker.is_delayed_submission_enabled() is enable_delayed_submission
        assert worker.is_python_future_enabled() is enable_python_future
