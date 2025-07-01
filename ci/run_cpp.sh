#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source "$(dirname "$0")/test_common.sh"

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

##################################### C++ ######################################
_SERVER_PORT=12345

run_cpp_tests() {
  RUNTIME_PATH=${CONDA_PREFIX:-./}
  BINARY_PATH=${RUNTIME_PATH}/bin

  # Disable memory get/put with RMM in protov1, it always segfaults.
  CMD_LINE="timeout 10m ${BINARY_PATH}/gtests/libucxx/UCXX_TEST --gtest_filter=-*RMM*Memory*"

  log_command "${CMD_LINE}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE}

  # Only test memory get/put with RMM in protov2, as protov1 segfaults.
  CMD_LINE="timeout 10m ${BINARY_PATH}/gtests/libucxx/UCXX_TEST --gtest_filter=*RMM*Memory*"

  log_command "${CMD_LINE}"
  UCX_PROTO_ENABLE=y UCX_TCP_CM_REUSEADDR=y ${CMD_LINE}
}

run_cpp_benchmark() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  RUNTIME_PATH=${CONDA_PREFIX:-./}
  BINARY_PATH=${RUNTIME_PATH}/bin

  CMD_LINE_SERVER="timeout 1m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE} -p ${SERVER_PORT}"
  CMD_LINE_CLIENT="timeout 1m ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -m ${PROGRESS_MODE} -p ${SERVER_PORT} 127.0.0.1"

  log_command "${CMD_LINE_SERVER}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE_SERVER} &
  sleep 1

  log_command "${CMD_LINE_CLIENT}"
  ${CMD_LINE_CLIENT}
}

run_cpp_example() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  RUNTIME_PATH=${CONDA_PREFIX:-./}
  BINARY_PATH=${RUNTIME_PATH}/bin

  CMD_LINE="timeout 1m ${BINARY_PATH}/examples/libucxx/ucxx_example_basic -m ${PROGRESS_MODE} -p ${SERVER_PORT}"

  log_command "${CMD_LINE}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE}
}

run_cpp_port_retry() {
  MAX_ATTEMPTS=${1}
  RUN_TYPE=${2}
  PROGRESS_MODE=${3}

  set +e
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    echo "Attempt ${attempt}/${MAX_ATTEMPTS} to run ${RUN_TYPE}"

    _SERVER_PORT=$((_SERVER_PORT + 1))    # Use different ports every time to prevent `Device is busy`

    if [[ "${RUN_TYPE}" == "benchmark" ]]; then
      run_cpp_benchmark ${_SERVER_PORT} "${PROGRESS_MODE}"
    elif [[ "${RUN_TYPE}" == "example" ]]; then
      run_cpp_example ${_SERVER_PORT} "${PROGRESS_MODE}"
    else
      set -e
      echo "Unknown test type ${RUN_TYPE}"
      exit 1
    fi

    LAST_STATUS=$?
    if [ ${LAST_STATUS} -eq 0 ]; then
      break;
    fi
    sleep 1
  done
  set -e

  if [ "${LAST_STATUS}" -ne 0 ]; then
    echo "Failure running benchmark client after ${MAX_ATTEMPTS} attempts"
    exit "$LAST_STATUS"
  fi
}

log_message "C++ Tests"
run_cpp_tests

log_message "C++ Benchmarks"
# run_cpp_port_retry MAX_ATTEMPTS RUN_TYPE PROGRESS_MODE
run_cpp_port_retry 10 "benchmark" "polling"
run_cpp_port_retry 10 "benchmark" "blocking"
run_cpp_port_retry 10 "benchmark" "thread-polling"
run_cpp_port_retry 10 "benchmark" "thread-blocking"
run_cpp_port_retry 10 "benchmark" "wait"

log_message "C++ Examples"
# run_cpp_port_retry MAX_ATTEMPTS RUN_TYPE PROGRESS_MODE
run_cpp_port_retry 10 "example" "polling"
run_cpp_port_retry 10 "example" "blocking"
run_cpp_port_retry 10 "example" "thread-polling"
run_cpp_port_retry 10 "example" "thread-blocking"
run_cpp_port_retry 10 "example" "wait"
