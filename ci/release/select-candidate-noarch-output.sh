#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 CONDA_OUTPUT_DIR HOST_ARCH CUDA_VERSION RELEASE_CANDIDATE" >&2
  exit 2
fi

output_dir="$1"
host_arch="$2"
cuda_version="$3"
release_candidate="$4"

case "${release_candidate}" in
  true)
    ;;
  false)
    exit 0
    ;;
  *)
    echo "release-candidate mode must be the literal true or false" >&2
    exit 1
    ;;
esac

if [[ ! "${cuda_version}" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]]; then
  echo "candidate noarch selection requires a concrete CUDA version" >&2
  exit 1
fi

if [[ ! -d "${output_dir}/noarch" ]]; then
  echo "candidate noarch package output directory is missing" >&2
  exit 1
fi

cuda_major="${cuda_version%%.*}"
mapfile -d '' distributed_packages < <(
  find "${output_dir}/noarch" -mindepth 1 -maxdepth 1 -type f \
    \( -name 'distributed-ucxx-*.conda' -o -name 'distributed-ucxx-*.tar.bz2' \) -print0
)

if [[ "${#distributed_packages[@]}" -ne 1 ]]; then
  echo "candidate build must produce exactly one distributed-ucxx noarch package" >&2
  exit 1
fi

package_path="${distributed_packages[0]}"
package_name="${package_path##*/}"
if [[ "${package_name}" != *"-cuda${cuda_major}_"* ]]; then
  echo "distributed-ucxx package build string does not match CUDA ${cuda_major}" >&2
  exit 1
fi

case "${host_arch}" in
  x86_64)
    ;;
  aarch64)
    rm -- "${package_path}"
    ;;
  *)
    echo "unsupported candidate host architecture: ${host_arch}" >&2
    exit 1
    ;;
esac
