#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2

# Clear out system ucx files to ensure that we're getting ucx from the wheel.
rm -rf /usr/lib64/ucx
rm -rf /usr/lib64/libuc*

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

librmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1529 cpp)
libucx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact ucx-wheels 1 cpp)

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

# Patch project metadata files to include the CUDA version suffix and version override.
pyproject_file="${package_dir}/pyproject.toml"

sed -i -E "s/^name = \"${package_name}(.*)?\"$/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
echo "${version}" > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_dir}/${package_name//-/_}/_version.py"

# For nightlies we want to ensure that we're pulling in alphas as well. The
# easiest way to do so is to augment the spec with a constraint containing a
# min alpha version that doesn't affect the version bounds but does allow usage
# of alpha versions for that dependency without --pre
alpha_spec=''
if ! rapids-is-release-build; then
    alpha_spec=',>=0.0.0a0'
fi

# Add -cuXX to package name
sed -r -i "s/rapids-dask-dependency==(.*)\"/rapids-dask-dependency==\1${alpha_spec}\"/g" ${pyproject_file}
sed -r -i "s/rmm(.*)\"/rmm${PACKAGE_CUDA_SUFFIX}\1${alpha_spec}\"/g" ${pyproject_file}
sed -r -i "s/cudf(.*)\"/cudf${PACKAGE_CUDA_SUFFIX}\1${alpha_spec}\"/g" ${pyproject_file}
sed -r -i "s/libucx(.*)\"/libucx${PACKAGE_CUDA_SUFFIX}\1${alpha_spec}\"/g" ${pyproject_file}

# Update cupy package name (different suffix from RAPIDS)
if [[ $PACKAGE_CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" ${pyproject_file}
fi

if [[ ${package_name} == "distributed-ucxx" ]]; then
    sed -r -i "s/\"ucxx(.*)\"/\"ucxx${PACKAGE_CUDA_SUFFIX}\1${alpha_spec}\"/g" ${pyproject_file}

    python -m pip wheel "${package_dir}/" -w "${package_dir}/dist" -vvv --no-deps --disable-pip-version-check

    RAPIDS_PY_WHEEL_NAME="distributed_ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/dist
elif [[ ${package_name} == "ucxx" ]]; then
    SKBUILD_CMAKE_ARGS="-DUCXX_ENABLE_RMM=ON" \
        python -m pip wheel "${package_dir}"/ -w "${package_dir}"/dist -vvv --no-deps --disable-pip-version-check --find-links ${librmm_wheelhouse} --find-links ${libucx_wheelhouse}

    python -m auditwheel repair -w ${package_dir}/final_dist --exclude "libucp.so.0" ${package_dir}/dist/*

    RAPIDS_PY_WHEEL_NAME="ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/final_dist
else
  echo "Unknown package '${package_name}'"
  exit 1
fi
