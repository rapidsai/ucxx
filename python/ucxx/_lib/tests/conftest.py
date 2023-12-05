# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import os

import pytest

# Prevent calls such as `cudf = pytest.importorskip("cudf")` from initializing
# a CUDA context. Such calls may cause tests that must initialize the CUDA
# context on the appropriate device to fail.
# For example, without `RAPIDS_NO_INITIALIZE=True`, `test_benchmark_cluster`
# will succeed if running alone, but fails when all tests are run in batch.
os.environ["RAPIDS_NO_INITIALIZE"] = "True"


@pytest.hookimpl()
def pytest_runtest_teardown(item, nextitem):
    gc.collect()
