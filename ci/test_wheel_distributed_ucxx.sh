#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source rapids-init-pip

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

rapids-logger "Run distributed-ucxx tests with wheels"
./ci/run_python_distributed.sh
