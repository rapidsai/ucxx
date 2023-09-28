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

_SERVER_PORT=12345

run_tests() {
  CMD_LINE="timeout 10m ${BINARY_PATH}/gtests/libucxx/UCXX_TEST"

  log_command "${CMD_LINE}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE}
}

run_benchmark() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  CMD_LINE_SERVER="timeout 1m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE} -p ${SERVER_PORT}"
  CMD_LINE_CLIENT="timeout 1m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE} -p ${SERVER_PORT} 127.0.0.1"

  log_command "${CMD_LINE_SERVER}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE_SERVER} &
  sleep 1

  log_command "${CMD_LINE_CLIENT}"
  ${CMD_LINE_CLIENT}
}

run_example() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  CMD_LINE="timeout 1m ${BINARY_PATH}/examples/libucxx/ucxx_example_basic -m ${PROGRESS_MODE} -p ${SERVER_PORT}"

  log_command "${CMD_LINE}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE}
}

run_port_retry() {
  MAX_ATTEMPTS=${1}
  RUN_TYPE=${2}
  PROGRESS_MODE=${3}

  set +e
  for attempt in $(seq 1 ${MAX_ATTEMPTS}); do
    echo "Attempt ${attempt}/${MAX_ATTEMPTS} to run ${RUN_TYPE}"

    _SERVER_PORT=$((_SERVER_PORT + 1))    # Use different ports every time to prevent `Device is busy`

    if [[ "${RUN_TYPE}" == "benchmark" ]]; then
      run_benchmark ${_SERVER_PORT} ${PROGRESS_MODE}
    elif [[ "${RUN_TYPE}" == "example" ]]; then
      run_example ${_SERVER_PORT} ${PROGRESS_MODE}
    else
      set -e
      echo "Unknown test type "${RUN_TYPE}""
      exit 1
    fi

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
# run_port_retry MAX_ATTEMPTS RUN_TYPE PROGRESS_MODE
run_port_retry 10 "benchmark" "polling"
run_port_retry 10 "benchmark" "blocking"
run_port_retry 10 "benchmark" "thread-polling"
run_port_retry 10 "benchmark" "thread-blocking"
run_port_retry 10 "benchmark" "wait"

rapids-logger "C++ Examples"
# run_port_retry MAX_ATTEMPTS RUN_TYPE PROGRESS_MODE
run_port_retry 10 "example" "polling"
run_port_retry 10 "example" "blocking"
run_port_retry 10 "example" "thread-polling"
run_port_retry 10 "example" "thread-blocking"
run_port_retry 10 "example" "wait"
