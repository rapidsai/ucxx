#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source "$(dirname "$0")/test_common.sh"

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test
conda activate test

rapids-print-env

print_system_stats

print_ucx_config

rapids-logger "Run tests with conda package"
rapids-logger "Python Core Tests"
# run_py_tests RUN_CYTHON
run_py_tests   1

rapids-logger "Python Async Tests"
# run_py_tests_async PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE SKIP
run_py_tests_async   thread          0                         0                    0
run_py_tests_async   thread          1                         1                    0
run_py_tests_async   blocking        0                         0                    0

rapids-logger "Python Benchmarks"
# run_py_benchmark  BACKEND   PROGRESS_MODE   ASYNCIO_WAIT  ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE NBUFFERS SLOW
run_py_benchmark    ucxx-core thread          0             0                         0                    1        0
run_py_benchmark    ucxx-core thread          1             0                         0                    1        0

for progress_mode in "blocking" "thread"; do
  for nbuf in 1 8; do
    # run_py_benchmark  BACKEND     PROGRESS_MODE     ASYNCIO_WAIT  ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE NBUFFERS SLOW
    run_py_benchmark    ucxx-async  ${progress_mode}  0             0                         0                    ${nbuf}  0
    run_py_benchmark    ucxx-async  ${progress_mode}  0             0                         1                    ${nbuf}  0
    if [[ ${progress_mode} != "blocking" ]]; then
      # Delayed submission isn't support by blocking progress mode
      # run_py_benchmark  BACKEND     PROGRESS_MODE     ASYNCIO_WAIT  ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE NBUFFERS SLOW
      run_py_benchmark    ucxx-async  ${progress_mode}  0             1                         0                    ${nbuf}  0
      run_py_benchmark    ucxx-async  ${progress_mode}  0             1                         1                    ${nbuf}  0
    fi
  done
done

rapids-logger "C++ future -> Python future notifier example"
timeout 1m python -m ucxx.examples.python_future_task_example
