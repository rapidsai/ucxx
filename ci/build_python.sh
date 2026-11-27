#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Runs ci/build_python.sh of the UCXX tree in the UCX repository, see ci/README.md.
# The workflow sources this script, so the UCXX one is sourced as well: the
# RAPIDS_PACKAGE_NAME it exports has to reach the workflow.

set -euo pipefail

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/ucx_bindings.sh"
source "${UCXX_SRC_DIR}/ci/build_python.sh"
