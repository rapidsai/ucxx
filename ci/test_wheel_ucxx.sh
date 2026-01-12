#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source rapids-init-pip

package_name="ucxx"

source "$(dirname "$0")/test_common.sh"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the packages built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
    ucxx_wheelhouse=$(rapids-download-from-github "$(rapids-package-name "wheel_python" ucxx --stable --cuda "$RAPIDS_CUDA_VERSION")")
    source ./ci/use_upstream_sabi_wheels.sh
else
    ucxx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
fi

libucxx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="libucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

rapids-pip-retry install \
    -v \
    "${libucxx_wheelhouse}/libucxx_${RAPIDS_PY_CUDA_SUFFIX}"*.whl \
    "$(echo "${ucxx_wheelhouse}"/"${package_name}_${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

rapids-logger "Run Python tests with wheels"
DISABLE_CYTHON=1 ./ci/run_python.sh
