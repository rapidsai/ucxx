#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source "$(dirname "$0")/test_common.sh"

conda activate test

rapids-print-env

print_system_stats

print_ucx_config

rapids-logger "Run C++ tests with conda package"
./ci/run_cpp.sh
