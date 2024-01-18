#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python"

./ci/build_wheel.sh ucxx ${package_dir}

RAPIDS_PY_WHEEL_NAME="ucxx_${AUDITWHEEL_POLICY}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/final_dist
