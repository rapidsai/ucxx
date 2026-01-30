#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX="true"
export RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX
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

rapids-logger "Run Python tests with wheels"
DISABLE_CYTHON=1 ./ci/run_python.sh
