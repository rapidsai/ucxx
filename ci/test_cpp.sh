#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source "$(dirname "$0")/test_common.sh"

conda activate test

rapids-print-env

print_system_stats

print_ucx_config

rapids-logger "Run tests with conda package"
rapids-logger "C++ Tests"
run_cpp_tests

rapids-logger "C++ Benchmarks"
# run_cpp_port_retry MAX_ATTEMPTS RUN_TYPE PROGRESS_MODE
run_cpp_port_retry 10 "benchmark" "polling"
run_cpp_port_retry 10 "benchmark" "blocking"
run_cpp_port_retry 10 "benchmark" "thread-polling"
run_cpp_port_retry 10 "benchmark" "thread-blocking"
run_cpp_port_retry 10 "benchmark" "wait"

rapids-logger "C++ Examples"
# run_cpp_port_retry MAX_ATTEMPTS RUN_TYPE PROGRESS_MODE
run_cpp_port_retry 10 "example" "polling"
run_cpp_port_retry 10 "example" "blocking"
run_cpp_port_retry 10 "example" "thread-polling"
run_cpp_port_retry 10 "example" "thread-blocking"
run_cpp_port_retry 10 "example" "wait"
