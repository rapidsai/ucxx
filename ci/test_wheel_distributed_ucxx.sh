#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

# TODO(jameslamb): revert before merging
git clone --branch generate-pip-constraints \
    https://github.com/rapidsai/gha-tools.git \
    /tmp/gha-tools

export PATH="/tmp/gha-tools/tools:${PATH}"

source rapids-init-pip

# TODO(jameslamb): revert before merging
source ci/use_wheels_from_prs.sh

package_name="distributed_ucxx"

source "$(dirname "$0")/test_common.sh"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

distributed_ucxx_wheelhouse=$(rapids-download-from-github "$(rapids-package-name wheel_python distributed-ucxx --pure --cuda "${RAPIDS_CUDA_VERSION}")")
ucxx_wheelhouse=$(rapids-download-from-github "$(rapids-package-name "wheel_python" ucxx --stable --cuda "$RAPIDS_CUDA_VERSION")")
libucxx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="libucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "${libucxx_wheelhouse}/libucxx_${RAPIDS_PY_CUDA_SUFFIX}"*.whl \
    "${ucxx_wheelhouse}/ucxx_${RAPIDS_PY_CUDA_SUFFIX}"*.whl \
    "$(echo "${distributed_ucxx_wheelhouse}"/"${package_name}_${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

rapids-logger "Run distributed-ucxx tests with wheels"
./ci/run_python_distributed.sh
