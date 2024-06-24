#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2

# Clear out system ucx files to ensure that we're getting ucx from the wheel.
rm -rf /usr/lib64/ucx
rm -rf /usr/lib64/libucm.*
rm -rf /usr/lib64/libucp.*
rm -rf /usr/lib64/libucs.*
rm -rf /usr/lib64/libucs_signal.*
rm -rf /usr/lib64/libuct.*

rm -rf /usr/include/ucm
rm -rf /usr/include/ucp
rm -rf /usr/include/ucs
rm -rf /usr/include/uct

source rapids-configure-sccache
source rapids-date-string

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

rapids-generate-version > ./VERSION

if [[ ${package_name} == "distributed-ucxx" ]]; then
    python -m pip wheel "${package_dir}/" -w "${package_dir}/dist" -vvv --no-deps --disable-pip-version-check

    RAPIDS_PY_WHEEL_NAME="distributed_ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/dist
elif [[ ${package_name} == "ucxx" ]]; then
    SKBUILD_CMAKE_ARGS="-DUCXX_ENABLE_RMM=ON" \
        python -m pip wheel "${package_dir}"/ -w "${package_dir}"/dist -vvv --no-deps --disable-pip-version-check

    python -m auditwheel repair -w ${package_dir}/final_dist --exclude "libucp.so.0" ${package_dir}/dist/*

    RAPIDS_PY_WHEEL_NAME="ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/final_dist
else
  echo "Unknown package '${package_name}'"
  exit 1
fi
