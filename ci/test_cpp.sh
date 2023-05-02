#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source "$(dirname "$0")/test_utils.sh"

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test
conda activate test

rapids-print-env

print_system_stats

BINARY_PATH=${CONDA_PREFIX}/bin

run_tests() {
  CMD_LINE="UCX_TCP_CM_REUSEADDR=y timeout 10m ${BINARY_PATH}/gtests/libucxx/UCXX_TEST --gtest_filter=-ListenerTest.CloseCallback:ListenerTest.IsAlive:ListenerTest.RaiseOnError"

  rapids-logger "Running: \n  - ${CMD_LINE}"

  UCX_TCP_CM_REUSEADDR=y timeout 10m ${BINARY_PATH}/gtests/libucxx/UCXX_TEST --gtest_filter=-ListenerTest.CloseCallback:ListenerTest.IsAlive:ListenerTest.RaiseOnError
}

run_benchmark() {
  PROGRESS_MODE=$1

  # UCX_TCP_CM_REUSEADDR=y to be able to bind immediately to the same port before
  # `TIME_WAIT` timeout
  CMD_LINE_SERVER="UCX_TCP_CM_REUSEADDR=y timeout 1m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE}"
  CMD_LINE_CLIENT="timeout 10m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE} 127.0.0.1"

  rapids-logger "Running: \n  - ${CMD_LINE_SERVER}\n  - ${CMD_LINE_CLIENT}"
  UCX_TCP_CM_REUSEADDR=y timeout 1m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE} &
  sleep 1
  timeout 1m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE} 127.0.0.1
}

run_example() {
  PROGRESS_MODE=$1

  # UCX_TCP_CM_REUSEADDR=y to be able to bind immediately to the same port before
  # `TIME_WAIT` timeout
  CMD_LINE="UCX_TCP_CM_REUSEADDR=y timeout 1m ${BINARY_PATH}/examples/libucxx/ucxx_example_basic -m ${PROGRESS_MODE}"

  rapids-logger "Running: \n  - ${CMD_LINE}"
  UCX_TCP_CM_REUSEADDR=y timeout 1m ${BINARY_PATH}/examples/libucxx/ucxx_example_basic -m ${PROGRESS_MODE}
}

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libucxx libucxx-examples libucxx-tests

print_ucx_config

rapids-logger "Run tests with conda package"
run_tests

# run_cpp_benchmark PROGRESS_MODE
run_benchmark   polling
run_benchmark   blocking
run_benchmark   thread-polling
run_benchmark   thread-blocking
run_benchmark   wait

# run_cpp_example PROGRESS_MODE
run_example   polling
run_example   blocking
run_example   thread-polling
run_example   thread-blocking
run_example   wait
