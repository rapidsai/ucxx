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

run_cpp_benchmark() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2
  TEST_TYPE=${3:-"tag_lat"}
  MEMORY_TYPE=${4:-"host"}
  MESSAGE_SIZE=${5:-"8388608"}
  NUM_ITERATIONS=${6:-"20"}
  NUM_WARMUP=${7:-"3"}
  PERCENTILE_RANK=${8:-"50.0"}
  ENDPOINT_ERROR_HANDLING=${9:-""}
  REUSE_ALLOCATIONS=${10:-""}
  VERIFY_RESULTS=${11:-""}

  RUNTIME_PATH=${CONDA_PREFIX:-./}
  BINARY_PATH=${RUNTIME_PATH}/bin

  # Build command line with all options
  CMD_BASE="${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest"
  CMD_OPTS="-t ${TEST_TYPE} -m ${MEMORY_TYPE} -P ${PROGRESS_MODE} -p ${SERVER_PORT} -s ${MESSAGE_SIZE} -n ${NUM_ITERATIONS} -w ${NUM_WARMUP} -R ${PERCENTILE_RANK}"

  # Add optional flags
  if [[ "${ENDPOINT_ERROR_HANDLING}" == "true" ]]; then
    CMD_OPTS="${CMD_OPTS} -e"
  fi
  if [[ "${REUSE_ALLOCATIONS}" == "false" ]]; then
    CMD_OPTS="${CMD_OPTS} -L"
  fi
  if [[ "${VERIFY_RESULTS}" == "true" ]]; then
    CMD_OPTS="${CMD_OPTS} -v"
  fi

  CMD_LINE_SERVER="timeout 1m ${CMD_BASE} ${CMD_OPTS}"
  CMD_LINE_CLIENT="timeout 1m ${CMD_BASE} ${CMD_OPTS} 127.0.0.1"

  log_command "${CMD_LINE_SERVER}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE_SERVER} &
  sleep 1

  log_command "${CMD_LINE_CLIENT}"
  ${CMD_LINE_CLIENT}
}

run_cpp_benchmark_comprehensive() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  # Test different combinations systematically

  # Basic latency test with default settings
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "50.0" "" "" ""

  # Test with different message sizes for latency
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8" "20" "3" "50.0" "" "" ""
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "1024" "20" "3" "50.0" "" "" ""
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "1048576" "20" "3" "50.0" "" "" ""

  # Test bandwidth test
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "host" "8388608" "20" "3" "50.0" "" "" ""

  # Test with different percentile ranks
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "95.0" "" "" ""
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "99.0" "" "" ""

  # Test with endpoint error handling
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "50.0" "true" "" ""

  # Test with disabled reuse allocations
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "50.0" "" "false" ""

  # Test with verification enabled
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "50.0" "" "" "true"

  # Test with different iteration counts
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "50" "5" "50.0" "" "" ""

  # Test with different warmup counts
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "10" "50.0" "" "" ""
}

run_cpp_benchmark_cuda_tests() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  # Only run CUDA tests if CUDA is available
  if command -v nvidia-smi &> /dev/null; then
    log_message "Running CUDA memory tests for progress mode: ${PROGRESS_MODE}"

    # Test CUDA memory types
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "cuda" "8388608" "20" "3" "50.0" "" "" ""
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "cuda" "8388608" "20" "3" "50.0" "" "" ""

    # Test CUDA managed memory
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "cuda-managed" "8388608" "20" "3" "50.0" "" "" ""
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "cuda-managed" "8388608" "20" "3" "50.0" "" "" ""

    # Test CUDA async memory
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "cuda-async" "8388608" "20" "3" "50.0" "" "" ""
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "cuda-async" "8388608" "20" "3" "50.0" "" "" ""

    # Test with smaller message sizes for CUDA
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "cuda" "1024" "20" "3" "50.0" "" "" ""
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "cuda-managed" "1024" "20" "3" "50.0" "" "" ""
  else
    log_message "Skipping CUDA tests - CUDA not available"
  fi
}

run_cpp_benchmark_edge_cases() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  # Test edge cases and combinations

  # Very small message size
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "1" "20" "3" "50.0" "" "" ""

  # Large message size
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "33554432" "10" "3" "50.0" "" "" ""

  # High percentile rank
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "99.9" "" "" ""

  # Many iterations
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "100" "10" "50.0" "" "" ""

  # All flags enabled
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "50.0" "true" "false" "true"

  # Bandwidth test with different sizes
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "host" "1048576" "20" "3" "50.0" "" "" ""
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "host" "16777216" "20" "3" "50.0" "" "" ""
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
    elif [[ "${RUN_TYPE}" == "benchmark_comprehensive" ]]; then
      run_cpp_benchmark_comprehensive ${_SERVER_PORT} "${PROGRESS_MODE}"
    elif [[ "${RUN_TYPE}" == "benchmark_cuda" ]]; then
      run_cpp_benchmark_cuda_tests ${_SERVER_PORT} "${PROGRESS_MODE}"
    elif [[ "${RUN_TYPE}" == "benchmark_edge_cases" ]]; then
      run_cpp_benchmark_edge_cases ${_SERVER_PORT} "${PROGRESS_MODE}"
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

log_message "C++ Benchmarks - Basic Tests"
# Basic tests for each progress mode
run_cpp_port_retry 10 "benchmark" "polling"
run_cpp_port_retry 10 "benchmark" "blocking"
run_cpp_port_retry 10 "benchmark" "thread-polling"
run_cpp_port_retry 10 "benchmark" "thread-blocking"
run_cpp_port_retry 10 "benchmark" "wait"

log_message "C++ Benchmarks - Comprehensive Tests"
# Comprehensive tests for a subset of progress modes (to avoid too many tests)
run_cpp_port_retry 10 "benchmark_comprehensive" "polling"
run_cpp_port_retry 10 "benchmark_comprehensive" "blocking"

log_message "C++ Benchmarks - CUDA Tests"
# CUDA tests for a subset of progress modes
run_cpp_port_retry 10 "benchmark_cuda" "polling"
run_cpp_port_retry 10 "benchmark_cuda" "blocking"

log_message "C++ Benchmarks - Edge Cases"
# Edge case tests for polling mode only (to keep test time reasonable)
run_cpp_port_retry 10 "benchmark_edge_cases" "polling"

log_message "C++ Examples"
# run_cpp_port_retry MAX_ATTEMPTS RUN_TYPE PROGRESS_MODE
run_cpp_port_retry 10 "example" "polling"
run_cpp_port_retry 10 "example" "blocking"
run_cpp_port_retry 10 "example" "thread-polling"
run_cpp_port_retry 10 "example" "thread-blocking"
run_cpp_port_retry 10 "example" "wait"
