#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-init-pip

package_name="ucxx"

source "$(dirname "$0")/test_common.sh"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

ucxx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
libucxx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="libucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

rapids-pip-retry install \
    -v \
    "${libucxx_wheelhouse}/libucxx_${RAPIDS_PY_CUDA_SUFFIX}"*.whl \
    "$(echo "${ucxx_wheelhouse}"/"${package_name}_${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

rapids-logger "Python Core Tests"
# run_py_tests RUN_CYTHON
run_py_tests   0

rapids-logger "Python Async Tests"
# run_py_tests_async PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE SKIP
run_py_tests_async   thread          1                         1                    0
