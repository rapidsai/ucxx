#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Runs the pre-commit checks of the UCXX tree in the UCX repository, see
# ci/README.md. pre-commit works from the root of a git repository and the
# UCXX hooks address their files from the UCXX root, so this uses the UCX
# pipeline's entry point, which runs them in a repository made of the UCXX
# tree alone.

set -euo pipefail

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/ucx_bindings.sh"
UCXX_DIR="${UCXX_SRC_DIR}" exec bash "${UCXX_UCX_SRC_DIR}/buildlib/tools/build_ucxx.sh" checks
