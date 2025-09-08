# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

# We try to be as close as possible to Distributed's testing, thus this file
# was taken from https://github.com/dask/distributed/blob/main/conftest.py,
# and minimal changes were applied.

# https://pytest.org/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
from __future__ import annotations

import pytest

# Force initialization of `rapids-dask-dependency` patches before importing
# `distributed_ucxx`. Without this, rewriting the `ucx://` prefix will
# fail to import `distributed_ucxx` as it causes a circular import.
#
# TODO: Remove once `rapids-dask-dependency` pins a `distributed` release containing
# https://github.com/dask/distributed/pull/9105
import distributed  # noqa: F401

try:
    import faulthandler
except ImportError:
    pass
else:
    try:
        faulthandler.enable()
    except Exception:
        pass

# Make all fixtures available
from distributed_ucxx.utils_test import *  # noqa


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

        if "ws" in item.fixturenames:
            item.add_marker(pytest.mark.workerstate)
