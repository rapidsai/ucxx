#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source "$(dirname "$0")/test_utils.sh"


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
  for attempt in $(seq 1 ${MAX_ATTEMPTS}); do
    echo "Attempt ${attempt}/${MAX_ATTEMPTS} to run ${RUN_TYPE}"

    _SERVER_PORT=$((_SERVER_PORT + 1))    # Use different ports every time to prevent `Device is busy`

    if [[ "${RUN_TYPE}" == "benchmark" ]]; then
      run_cpp_benchmark ${_SERVER_PORT} ${PROGRESS_MODE}
    elif [[ "${RUN_TYPE}" == "example" ]]; then
      run_cpp_example ${_SERVER_PORT} ${PROGRESS_MODE}
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


#################################### Python ####################################
run_py_tests() {
  CMD_LINE="timeout 2m python -m pytest -vs python/ucxx/_lib/tests/"
  log_command "${CMD_LINE}"
  timeout 2m python -m pytest -vs python/ucxx/_lib/tests/
}

run_py_tests_async() {
  PROGRESS_MODE=$1
  ENABLE_DELAYED_SUBMISSION=$2
  ENABLE_PYTHON_FUTURE=$3
  SKIP=$4

  CMD_LINE="UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} timeout 20m python -m pytest -vs python/ucxx/_lib_async/tests/ --durations=50"

  if [ $SKIP -ne 0 ]; then
    echo -e "\e[1;33mSkipping unstable test: ${CMD_LINE}\e[0m"
  else
    log_command "${CMD_LINE}"
    UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} timeout 20m python -m pytest -vs python/ucxx/_lib_async/tests/ --durations=50
  fi
}

run_py_benchmark() {
  BACKEND=$1
  PROGRESS_MODE=$2
  ASYNCIO_WAIT=$3
  ENABLE_DELAYED_SUBMISSION=$4
  ENABLE_PYTHON_FUTURE=$5
  N_BUFFERS=$6
  SLOW=$7

  if [ $ASYNCIO_WAIT -ne 0 ]; then
    ASYNCIO_WAIT="--asyncio-wait"
  else
    ASYNCIO_WAIT=""
  fi

  CMD_LINE="UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} timeout 2m python -m ucxx.benchmarks.send_recv --backend ${BACKEND} -o cupy --reuse-alloc -n 8MiB --n-buffers $N_BUFFERS --progress-mode ${PROGRESS_MODE} ${ASYNCIO_WAIT}"

  # Workaround for https://github.com/rapidsai/ucxx/issues/15
  CMD_LINE="UCX_KEEPALIVE_INTERVAL=1ms ${CMD_LINE}"

  log_command "${CMD_LINE}"
  if [ $SLOW -ne 0 ]; then
    echo -e "\e[1;33mSLOW BENCHMARK: it may seem like a deadlock but will eventually complete.\e[0m"
  fi

  UCX_KEEPALIVE_INTERVAL=1ms UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} timeout 2m python -m ucxx.benchmarks.send_recv --backend ${BACKEND} -o cupy --reuse-alloc -n 8MiB --n-buffers $N_BUFFERS --progress-mode ${PROGRESS_MODE} ${ASYNCIO_WAIT}
}

################################## Distributed #################################
run_distributed_ucxx_tests() {
  PROGRESS_MODE=$1
  ENABLE_DELAYED_SUBMISSION=$2
  ENABLE_PYTHON_FUTURE=$3

  CMD_LINE="UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} timeout 10m python -m pytest -vs python/distributed-ucxx/distributed_ucxx/tests/"

  # Workaround for https://github.com/rapidsai/ucxx/issues/15
  # CMD_LINE="UCX_KEEPALIVE_INTERVAL=1ms ${CMD_LINE}"

  log_command "${CMD_LINE}"
  UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} timeout 10m python -m pytest -vs python/distributed-ucxx/distributed_ucxx/tests/
}