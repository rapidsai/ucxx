#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source "$(dirname "$0")/test_common.sh"

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 python)

CUDF_CPP_CHANNEL=$(rapids-get-pr-conda-artifact cudf 16806 cpp)
CUDF_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact cudf 16806 python)

rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --prepend-channel "${LIBRMM_CHANNEL}" \
  --prepend-channel "${RMM_CHANNEL}" \
  --prepend-channel "${CUDF_CPP_CHANNEL}" \
  --prepend-channel "${CUDF_PYTHON_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test
conda activate test

rapids-print-env

print_system_stats

BINARY_PATH=${CONDA_PREFIX}/bin

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${CUDF_CPP_CHANNEL}" \
  --channel "${CUDF_PYTHON_CHANNEL}" \
  libucxx libucxx-examples libucxx-tests

print_ucx_config

rapids-logger "Run tests with conda package"
rapids-logger "C++ Tests"
run_cpp_tests

rapids-logger "C++ Benchmarks"
# run_cpp_port_retry MAX_ATTEMPTS RUN_TYPE PROGRESS_MODE
run_cpp_port_retry 10 "benchmark" "polling"
run_cpp_port_retry 10 "benchmark" "blocking"
run_cpp_port_retry 10 "benchmark" "thread-polling"
run_cpp_port_retry 10 "benchmark" "thread-blocking"
run_cpp_port_retry 10 "benchmark" "wait"

rapids-logger "C++ Examples"
# run_cpp_port_retry MAX_ATTEMPTS RUN_TYPE PROGRESS_MODE
run_cpp_port_retry 10 "example" "polling"
run_cpp_port_retry 10 "example" "blocking"
run_cpp_port_retry 10 "example" "thread-polling"
run_cpp_port_retry 10 "example" "thread-blocking"
run_cpp_port_retry 10 "example" "wait"
