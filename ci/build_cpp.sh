#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin C++ and Python builds"

sccache --zero-stats

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1739 cpp)

rapids-conda-retry mambabuild \
  conda/recipes/ucxx \
  --channel "${LIBRMM_CHANNEL}"

sccache --show-adv-stats

rapids-upload-conda-to-s3 cpp
