#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python"

export RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

./ci/build_wheel.sh ucxx ${package_dir}

RAPIDS_PY_WHEEL_NAME="ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/final_dist
