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

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 python)
CUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 16806 python)
LIBCUDF_CHANNEL=$(rapids-get-pr-conda-artifact libcudf 16806 cpp)
PYLIBCUDF_CHANNEL=$(rapids-get-pr-conda-artifact pycudf 16806 python)
rapids-conda-retry mambabuild \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${PYLIBCUDF_CHANNEL}" \
  conda/recipes/ucxx

rapids-upload-conda-to-s3 cpp
