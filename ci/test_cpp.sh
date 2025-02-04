#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source ./ci/use_gha_tools_from_branch.sh

source "$(dirname "$0")/test_common.sh"

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

UCXX_VERSION="$(head -1 ./VERSION)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
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
  "libucxx=${UCXX_VERSION}" \
  "libucxx-examples=${UCXX_VERSION}" \
  "libucxx-tests=${UCXX_VERSION}"

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
