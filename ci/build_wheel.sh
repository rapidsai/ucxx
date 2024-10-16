#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
package_type=$3

# The list of shared libraries to exclude from wheels varies by project.
#
# Capturing that here in argument-parsing to allow this build_wheel.sh
# script to be re-used by all wheel builds in the project.
case "${package_dir}" in
  python/libucxx)
    EXCLUDE_ARGS=(
      --exclude "libucp.so.0"
    )
  ;;
  python/ucxx)
    EXCLUDE_ARGS=(
      --exclude "libucp.so.0"
      --exclude "libucxx.so"
    )
  ;;
  *)
    EXCLUDE_ARGS=()
  ;;
esac

source rapids-configure-sccache
source rapids-date-string

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

rapids-generate-version > ./VERSION

cd "${package_dir}"

rapids-logger "Building '${package_name}' wheel"
python -m pip wheel \
    -w dist \
    -v \
    --no-build-isolation \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats

mkdir -p final_dist
python -m auditwheel repair \
    "${EXCLUDE_ARGS[@]}" \
    -w final_dist \
    dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 "${package_type}" final_dist
