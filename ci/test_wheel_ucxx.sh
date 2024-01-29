#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

PROJECT_NAME="ucxx"

source "$(dirname "$0")/test_utils.sh"
source "$(dirname "$0")/test_common.sh"

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${PROJECT_NAME}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/${PROJECT_NAME}*.whl)[test]

print_ucx_config

# Run smoke tests for aarch64 pull requests
if [[ "$(arch)" == "aarch64" && "${RAPIDS_BUILD_TYPE}" == "pull-request" ]]; then
  rapids-logger "Python Async Smoke Tests"
  pytest -vs ci/wheel_smoke_test.py
else
  rapids-logger "Python Core Tests"
  run_py_tests

  rapids-logger "Python Async Tests"
  # run_py_tests_async PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE SKIP
  run_py_tests_async   thread          1                         1                    0
fi
