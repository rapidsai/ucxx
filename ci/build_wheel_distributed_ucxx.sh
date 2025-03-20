#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/distributed-ucxx"
wheel_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR}

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

./ci/build_wheel.sh distributed-ucxx "${package_dir}"
mkdir -p "${wheel_dir}"
cp "${package_dir}/dist"/* "${wheel_dir}/"
./ci/validate_wheel.sh "${package_dir}" "${wheel_dir}"

RAPIDS_PY_WHEEL_NAME="distributed_ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python "${wheel_dir}"
