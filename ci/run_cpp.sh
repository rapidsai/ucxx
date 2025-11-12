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

  CMD_LINE="timeout 10m ${BINARY_PATH}/gtests/libucxx/UCXX_TEST"

  log_command "${CMD_LINE}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE}
}

run_cpp_benchmarks() {
  # Run the dedicated benchmark script
  "$(dirname "$0")/run_cpp_benchmarks.sh"
}

run_cpp_example() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  RUNTIME_PATH=${CONDA_PREFIX:-./}
  BINARY_PATH=${RUNTIME_PATH}/bin

  CMD_LINE="timeout 1m ${BINARY_PATH}/examples/libucxx/ucxx_example_basic -P ${PROGRESS_MODE} -p ${SERVER_PORT}"

  log_command "${CMD_LINE}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE}
}

log_message "C++ Tests"
run_cpp_tests

log_message "C++ Examples"
# run_port_retry MAX_ATTEMPTS RUN_TYPE PROGRESS_MODE FUNCTION_NAME
run_port_retry 10 "example" "polling" "run_cpp_example"
run_port_retry 10 "example" "blocking" "run_cpp_example"
run_port_retry 10 "example" "thread-polling" "run_cpp_example"
run_port_retry 10 "example" "thread-blocking" "run_cpp_example"
run_port_retry 10 "example" "wait" "run_cpp_example"
