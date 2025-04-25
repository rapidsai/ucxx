#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="distributed_ucxx"

source "$(dirname "$0")/test_common.sh"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

distributed_ucxx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
ucxx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
libucxx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="libucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

rapids-pip-retry install \
    -v \
    "${libucxx_wheelhouse}/libucxx_${RAPIDS_PY_CUDA_SUFFIX}"*.whl \
    "${ucxx_wheelhouse}/ucxx_${RAPIDS_PY_CUDA_SUFFIX}"*.whl \
    "$(echo "${distributed_ucxx_wheelhouse}"/"${package_name}_${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

rapids-logger "Distributed Tests"

# run_distributed_ucxx_tests    PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION   ENABLE_PYTHON_FUTURE
run_distributed_ucxx_tests      thread          1                           1

install_distributed_dev_mode

# run_distributed_ucxx_tests_internal PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION   ENABLE_PYTHON_FUTURE
run_distributed_ucxx_tests_internal   thread          1                           1
