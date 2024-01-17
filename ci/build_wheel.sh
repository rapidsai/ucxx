#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="ucxx"
package_dir="python"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

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

# add or replace -cuXX to package name. Determine if this is a package mention based
# on whether it has any sort of pinning. There's a usage of "rmm" in known_rapids that
# should not get this -cuXX tacked on.
sed -r -i -E "s/rmm(-cu[0-9]+|)([\=\<\>\!].*)/rmm${PACKAGE_CUDA_SUFFIX}\2/g" ${pyproject_file}
# Capture the pin and add the alpha spec to it as needed
sed -r -i -E "/${alpha_spec}\"(,|)$/! s/rmm-cu[0-9]+([\=\<\>\!].*)\"/rmm${PACKAGE_CUDA_SUFFIX}\1${alpha_spec}\"/g" ${pyproject_file}

cd "${package_dir}"

SKBUILD_CMAKE_ARGS="-DUCXX_ENABLE_PYTHON=ON -DUCXX_ENABLE_RMM=ON" \
    python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

python -m auditwheel repair -w ${package_dir}/final_dist ${package_dir}/dist/*

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="ucxx_${AUDITWHEEL_POLICY}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/final_dist
