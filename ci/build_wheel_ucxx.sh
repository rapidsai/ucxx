#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/ucxx"

# Downloads libucxx wheel from this current build,
# then ensures 'ucxx' wheel builds always use the 'libucxx' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
RAPIDS_PY_WHEEL_NAME="libucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libucxx_dist
echo "libucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libucxx_dist/libucxx_*.whl)" > /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

export SKBUILD_CMAKE_ARGS="-DFIND_UCXX_CPP=ON;-DCMAKE_INSTALL_LIBDIR=ucxx/lib64;-DCMAKE_INSTALL_INCLUDEDIR=ucxx/include"

./ci/build_wheel.sh ucxx "${package_dir}" python
