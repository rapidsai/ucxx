#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="ucxx"
package_dir="python/ucxx"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libucxx wheel from this current build,
# then ensures 'ucxx' wheel builds always use the 'libucxx' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
LIBUCXX_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
echo "libucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBUCXX_WHEELHOUSE}"/libucxx_*.whl)" > /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

export SKBUILD_CMAKE_ARGS="-DFIND_UCXX_CPP=ON;-DCMAKE_INSTALL_LIBDIR=ucxx/lib64;-DCMAKE_INSTALL_INCLUDEDIR=ucxx/include"

./ci/build_wheel.sh "${package_name}" "${package_dir}"

python -m auditwheel repair \
    --exclude "libucp.so.0" \
    --exclude "libucxx.so" \
    --exclude "librapids_logger.so" \
    --exclude "librmm.so" \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
