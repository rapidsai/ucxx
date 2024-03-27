#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

PROJECT_NAME="distributed_ucxx"

source "$(dirname "$0")/test_common.sh"

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${PROJECT_NAME}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# Install previously built ucxx wheel
RAPIDS_PY_WHEEL_NAME="ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-ucxx-dep
python -m pip install ./local-ucxx-dep/ucxx*.whl

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/${PROJECT_NAME}*.whl)[test]

rapids-logger "Distributed Tests"

# run_distributed_ucxx_tests    PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION   ENABLE_PYTHON_FUTURE
run_distributed_ucxx_tests      thread          1                           1

install_distributed_dev_mode

# run_distributed_ucxx_tests_internal PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION   ENABLE_PYTHON_FUTURE
run_distributed_ucxx_tests_internal   thread          1                           1
