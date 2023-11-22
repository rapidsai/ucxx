#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1374 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1374 python)

rapids-logger "Begin C++ and Python builds"

rapids-conda-retry mambabuild \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  conda/recipes/ucxx

rapids-upload-conda-to-s3 cpp
