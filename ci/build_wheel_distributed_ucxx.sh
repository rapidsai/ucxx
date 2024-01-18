#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/distributed_ucxx"

./ci/build_wheel.sh dask-cudf ${package_dir}

RAPIDS_PY_WHEEL_NAME="distributed_ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/dist
