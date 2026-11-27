#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Replaces the UCXX sources of this checkout with the ones of the UCXX tree in
# the UCX repository, for the devcontainer build, which builds this checkout in
# place; see ci/README.md. python/distributed-ucxx and the build configuration
# (dependencies.yaml, .devcontainer) stay as they are. Symlinks are resolved:
# some point outside of the UCXX tree, into the UCX repository.

set -euo pipefail

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/ucx_bindings.sh"
for path in cpp python/libucxx python/ucxx; do
  rm -rf "${UCXX_REPO_DIR:?}/${path}"
  cp -aL "${UCXX_SRC_DIR}/${path}" "${UCXX_REPO_DIR}/${path}"
done
echo "Replaced cpp, python/libucxx and python/ucxx with ${UCXX_SRC_DIR}"
