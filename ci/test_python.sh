#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source "$(dirname "$0")/test_utils.sh"

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test
conda activate test

rapids-print-env

print_system_stats

run_tests() {
  rapids-logger "Running: timeout 2m pytest-vs python/ucxx/_lib/tests/"
  timeout 2m pytest -vs python/ucxx/_lib/tests/
}

run_tests_async() {
  PROGRESS_MODE=$1
  ENABLE_DELAYED_SUBMISSION=$2
  ENABLE_PYTHON_FUTURE=$3
  SKIP=$4

  CMD_LINE="UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} timeout 20m pytest -vs python/ucxx/_lib_async/tests/ --durations=50"

  if [ $SKIP -ne 0 ]; then
    rapids-logger "Skipping unstable test: ${CMD_LINE}"
  else
    rapids-logger "Running: ${CMD_LINE}"
    UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} timeout 20m pytest -vs python/ucxx/_lib_async/tests/ --durations=50
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

  rapids-logger "Running: ${CMD_LINE}"
  if [ $SLOW -ne 0 ]; then
    rapids-logger "SLOW BENCHMARK: it may seem like a deadlock but will eventually complete."
  fi

  UCX_KEEPALIVE_INTERVAL=1ms UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} timeout 2m python -m ucxx.benchmarks.send_recv --backend ${BACKEND} -o cupy --reuse-alloc -n 8MiB --n-buffers $N_BUFFERS --progress-mode ${PROGRESS_MODE} ${ASYNCIO_WAIT}
}

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=${RAPIDS_CONDA_BLD_OUTPUT_DIR}

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libucxx ucxx

rapids-logger "Run tests with conda package"
run_tests

print_ucx_config

rapids-logger "\e[1mRunning: pytest-vs python/ucxx/_lib/tests/\e[0m"
pytest -vs python/ucxx/_lib/tests/

# run_tests_async PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE SKIP
run_tests_async   thread          0                         0                    0
run_tests_async   thread          1                         1                    0

# run_py_benchmark  BACKEND   PROGRESS_MODE   ASYNCIO_WAIT  ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE NBUFFERS SLOW
run_py_benchmark    ucxx-core thread          0             0                         0                    1        0
run_py_benchmark    ucxx-core thread          1             0                         0                    1        0

for nbuf in 1 8; do
  if [[ ! $RAPIDS_CUDA_VERSION =~ 11.2.* ]]; then
    # run_py_benchmark  BACKEND     PROGRESS_MODE   ASYNCIO_WAIT  ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE NBUFFERS SLOW
    run_py_benchmark    ucxx-async  thread          0             0                         0                    ${nbuf}  0
    run_py_benchmark    ucxx-async  thread          0             0                         1                    ${nbuf}  0
    run_py_benchmark    ucxx-async  thread          0             1                         0                    ${nbuf}  0
    run_py_benchmark    ucxx-async  thread          0             1                         1                    ${nbuf}  0
  fi
done
