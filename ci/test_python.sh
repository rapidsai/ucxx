#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source ./test_utils.sh

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

run_tests_async() {
  PROGRESS_MODE=$1
  ENABLE_DELAYED_SUBMISSION=$2
  ENABLE_PYTHON_FUTURE=$3
  SKIP=$4

  CMD_LINE="UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} pytest -vs python/ucxx/_lib_async/tests/"

  if [ $SKIP -ne 0 ]; then
    rapids-logger "\e[31;1mSkipping unstable test: ${CMD_LINE}\e[0m"
  else
    rapids-logger "\e[1mRunning: ${CMD_LINE}\e[0m"
    UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} pytest -vs python/ucxx/_lib_async/tests/
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

  CMD_LINE="UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} python -m ucxx.benchmarks.send_recv --backend ${BACKEND} -o cupy --reuse-alloc -d 0 -e 1 -n 8MiB --n-buffers $N_BUFFERS --progress-mode ${PROGRESS_MODE} ${ASYNCIO_WAIT}"

  rapids-logger "\e[1mRunning: ${CMD_LINE}\e[0m"
  if [ $SLOW -ne 0 ]; then
    rapids-logger "\e[31;1mSLOW BENCHMARK: it may seem like a deadlock but will eventually complete.\e[0m"
  fi
  UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} python -m ucxx.benchmarks.send_recv --backend ${BACKEND} -o cupy --reuse-alloc -d 0 -e 1 -n 8MiB --n-buffers $N_BUFFERS --progress-mode ${PROGRESS_MODE} ${ASYNCIO_WAIT}
}

rapids-logger "Downloading artifacts from previous jobs"
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  --channel "${PYTHON_CHANNEL}" \
  ucxx

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
  # run_py_benchmark  BACKEND     PROGRESS_MODE   ASYNCIO_WAIT  ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE NBUFFERS SLOW
  run_py_benchmark    ucxx-async  thread          0             0                         0                    ${nbuf}  0
  run_py_benchmark    ucxx-async  thread          0             0                         1                    ${nbuf}  0
  run_py_benchmark    ucxx-async  thread          0             1                         0                    ${nbuf}  0
  run_py_benchmark    ucxx-async  thread          0             1                         1                    ${nbuf}  0
done
