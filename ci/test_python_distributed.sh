#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source "$(dirname "$0")/test_common.sh"
source "$(dirname "$0")/use_conda_packages_from_prs.sh"

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key test_python_distributed \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test
conda activate test

rapids-print-env

print_system_stats

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libucxx ucxx distributed-ucxx

print_ucx_config

rapids-logger "Run distributed-ucxx tests with conda package"
# run_distributed_ucxx_tests    PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION   ENABLE_PYTHON_FUTURE
run_distributed_ucxx_tests      polling         0                           0
run_distributed_ucxx_tests      thread          0                           0
run_distributed_ucxx_tests      thread          0                           1
run_distributed_ucxx_tests      thread          1                           0
run_distributed_ucxx_tests      thread          1                           1

install_distributed_dev_mode

# run_distributed_ucxx_tests_internal   PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION   ENABLE_PYTHON_FUTURE
run_distributed_ucxx_tests_internal     polling         0                           0
run_distributed_ucxx_tests_internal     thread          0                           0
run_distributed_ucxx_tests_internal     thread          0                           1
run_distributed_ucxx_tests_internal     thread          1                           0
run_distributed_ucxx_tests_internal     thread          1                           1
