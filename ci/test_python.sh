#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source "$(dirname "$0")/test_common.sh"

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

UCXX_VERSION="$(rapids-version)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test
conda activate test

rapids-print-env

print_system_stats

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  "libucxx=${UCXX_VERSION}" \
  "ucxx=${UCXX_VERSION}"

print_ucx_config

rapids-logger "Run tests with conda package"
rapids-logger "Python Core Tests"
run_py_tests

rapids-logger "Python Async Tests"
# run_py_tests_async PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE SKIP
run_py_tests_async   thread          0                         0                    0
run_py_tests_async   thread          1                         1                    0

rapids-logger "Python Benchmarks"
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

rapids-logger "C++ future -> Python future notifier example"
timeout 1m python -m ucxx.examples.python_future_task_example
