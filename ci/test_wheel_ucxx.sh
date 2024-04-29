#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="ucxx"

source "$(dirname "$0")/test_common.sh"

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

libucx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact ucx-wheels 1 cpp)

python -m pip install "${package_name}-${RAPIDS_PY_CUDA_SUFFIX}[test]" --find-links dist/ --find-links "${libucx_wheelhouse}"

rapids-logger "Python Core Tests"
run_py_tests

rapids-logger "Python Async Tests"
# run_py_tests_async PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE SKIP
run_py_tests_async   thread          1                         1                    0
