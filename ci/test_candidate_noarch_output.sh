#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
selector="${repository_root}/ci/release/select-candidate-noarch-output.sh"
ucxx_version="$(tr -d '\r\n' < "${repository_root}/VERSION")"
ucxx_package_version="${ucxx_version%.*}"
temporary_directory="$(mktemp -d)"
trap 'rm -rf "${temporary_directory}"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

new_output_dir() {
  local name="$1"
  mkdir -p "${temporary_directory}/${name}/noarch" "${temporary_directory}/${name}/linux-64"
  touch "${temporary_directory}/${name}/noarch/ucxx-${ucxx_package_version}-cuda12_example.conda"
  touch "${temporary_directory}/${name}/linux-64/ucxx-${ucxx_package_version}-cuda12_example.conda"
}

for cuda_version in 12.9.2 13.3.0; do
  cuda_major="${cuda_version%%.*}"
  for host_arch in x86_64 aarch64; do
    case "${host_arch}" in
      x86_64) package_arch="amd64" ;;
      aarch64) package_arch="arm64" ;;
    esac
    output_name="cuda${cuda_major}-${package_arch}"
    new_output_dir "${output_name}"
    output_dir="${temporary_directory}/${output_name}"
    package_path="${output_dir}/noarch/distributed-ucxx-${ucxx_package_version}-cuda${cuda_major}_261015053754_293362d5_h17ac89a.conda"
    touch "${package_path}"
    "${selector}" "${output_dir}" "${host_arch}" "${cuda_version}" true
    if [[ "${host_arch}" == x86_64 ]]; then
      [[ -f "${package_path}" ]] || fail "canonical amd64 must retain CUDA ${cuda_major} package"
    else
      [[ ! -e "${package_path}" ]] || fail "arm64 candidate must not export duplicate CUDA ${cuda_major} noarch package"
    fi
    [[ -f "${output_dir}/linux-64/ucxx-${ucxx_package_version}-cuda12_example.conda" ]] || fail "candidate selection changed a native package"
    [[ -f "${output_dir}/noarch/ucxx-${ucxx_package_version}-cuda12_example.conda" ]] || fail "candidate selection changed another noarch package"
  done
done

new_output_dir ordinary-arm64
ordinary_arm64="${temporary_directory}/ordinary-arm64"
ordinary_package="${ordinary_arm64}/noarch/distributed-ucxx-${ucxx_package_version}-cuda12_261015053754_293362d5_h17ac89a.conda"
touch "${ordinary_package}"
"${selector}" "${ordinary_arm64}" aarch64 not-a-version false
[[ -f "${ordinary_package}" ]] || fail "ordinary builds must retain the existing package output"

new_output_dir missing-package
if "${selector}" "${temporary_directory}/missing-package" x86_64 12.9.2 true >/dev/null 2>&1; then
  fail "candidate build without the expected package must fail"
fi

new_output_dir missing-cuda
missing_cuda_package="${temporary_directory}/missing-cuda/noarch/distributed-ucxx-${ucxx_package_version}-cuda12_261015053754_293362d5_h17ac89a.conda"
touch "${missing_cuda_package}"
if "${selector}" "${temporary_directory}/missing-cuda" x86_64 "" true >/dev/null 2>&1; then
  fail "candidate build without a CUDA version must fail"
fi

new_output_dir unsupported-arch
unsupported_package="${temporary_directory}/unsupported-arch/noarch/distributed-ucxx-${ucxx_package_version}-cuda12_261015053754_293362d5_h17ac89a.conda"
touch "${unsupported_package}"
if "${selector}" "${temporary_directory}/unsupported-arch" armv7l 12.9.2 true >/dev/null 2>&1; then
  fail "unsupported candidate host architecture must fail"
fi
[[ -f "${unsupported_package}" ]] || fail "unsupported architecture failure must not remove output"

new_output_dir mismatched-cuda
mismatched_package="${temporary_directory}/mismatched-cuda/noarch/distributed-ucxx-${ucxx_package_version}-cuda13_261015053754_293362d5_h17ac89a.conda"
touch "${mismatched_package}"
if "${selector}" "${temporary_directory}/mismatched-cuda" x86_64 12.9.2 true >/dev/null 2>&1; then
  fail "candidate package CUDA build string must match the matrix CUDA major"
fi

new_output_dir duplicate-package
touch "${temporary_directory}/duplicate-package/noarch/distributed-ucxx-${ucxx_package_version}-cuda12_261015053754_293362d5_h17ac89a.conda"
touch "${temporary_directory}/duplicate-package/noarch/distributed-ucxx-${ucxx_package_version}-cuda12_261015053754_293362d5_h17ac89a.tar.bz2"
if "${selector}" "${temporary_directory}/duplicate-package" x86_64 12.9.2 true >/dev/null 2>&1; then
  fail "candidate build with ambiguous noarch package outputs must fail"
fi

echo "Candidate noarch package selection checks passed"
