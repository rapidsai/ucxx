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

NEXT_PORT=12345
function get_next_port() {
  echo ${NEXT_PORT}
  NEXT_PORT=$((NEXT_PORT + 1))
}

run_tests() {
  CMD_LINE="UCX_TCP_CM_REUSEADDR=y timeout 10m ${BINARY_PATH}/gtests/libucxx/UCXX_TEST"

  log_command "${CMD_LINE}"

  UCX_TCP_CM_REUSEADDR=y timeout 10m ${BINARY_PATH}/gtests/libucxx/UCXX_TEST
}

run_benchmark() {
  PROGRESS_MODE=$1

  MAX_ATTEMPTS=10

  set +e
  for attempt in $(seq 1 ${MAX_ATTEMPTS}); do
    echo "Attempt ${attempt}/${MAX_ATTEMPTS} to run benchmark"

    SERVER_PORT=$(get_next_port)    # Use different ports every time to prevent `Device is busy`

    CMD_LINE_SERVER="timeout 1m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE} -p ${SERVER_PORT} &"
    CMD_LINE_CLIENT="timeout 1m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE} -p ${SERVER_PORT} 127.0.0.1"

    log_command "${CMD_LINE_SERVER}"
    log_command "${CMD_LINE_CLIENT}"
    UCX_TCP_CM_REUSEADDR=y timeout 1m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE} -p ${SERVER_PORT} &
    sleep 1

    timeout 1m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE} -p ${SERVER_PORT} 127.0.0.1
    LAST_STATUS=$?
    if [ ${LAST_STATUS} -eq 0 ]; then
      break;
    fi
    sleep 1
  done
  set -e

  if [ ${LAST_STATUS} -ne 0 ]; then
    echo "Failure running benchmark client after ${MAX_ATTEMPTS} attempts"
    exit $LAST_STATUS
  fi
}

run_example() {
  PROGRESS_MODE=$1

  SERVER_PORT=$(get_next_port)    # Use different ports every time to prevent `Device is busy`

  CMD_LINE="timeout 1m ${BINARY_PATH}/examples/libucxx/ucxx_example_basic -m ${PROGRESS_MODE} -p ${SERVER_PORT}"

  log_command "${CMD_LINE}"
  UCX_TCP_CM_REUSEADDR=y timeout 1m ${BINARY_PATH}/examples/libucxx/ucxx_example_basic -m ${PROGRESS_MODE} -p ${SERVER_PORT}
}

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libucxx libucxx-examples libucxx-tests

print_ucx_config

rapids-logger "Run tests with conda package"
rapids-logger "C++ Tests"
run_tests

rapids-logger "C++ Benchmarks"
# run_cpp_benchmark PROGRESS_MODE
run_benchmark   polling
run_benchmark   blocking
run_benchmark   thread-polling
run_benchmark   thread-blocking
run_benchmark   wait

rapids-logger "C++ Examples"
# run_cpp_example PROGRESS_MODE
run_example   polling
run_example   blocking
run_example   thread-polling
run_example   thread-blocking
run_example   wait
