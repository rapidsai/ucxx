#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Runs ci/build_docs.sh of the UCXX tree in the UCX repository, see ci/README.md.

set -euo pipefail

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/ucx_bindings.sh"
exec "${UCXX_SRC_DIR}/ci/build_docs.sh" "$@"
