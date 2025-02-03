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

LIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 cpp conda)
PYLIBRMM_CHANNEL10=$(RAPIDS_PY_VERSION=3.10 _rapids-get-pr-artifact rmm 1808 python conda)
PYLIBRMM_CHANNEL11=$(RAPIDS_PY_VERSION=3.11 _rapids-get-pr-artifact rmm 1808 python conda)
PYLIBRMM_CHANNEL12=$(RAPIDS_PY_VERSION=3.12 _rapids-get-pr-artifact rmm 1808 python conda)

rapids-conda-retry mambabuild \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${PYLIBRMM_CHANNEL10}" \
  --channel "${PYLIBRMM_CHANNEL11}" \
  --channel "${PYLIBRMM_CHANNEL12}" \
  conda/recipes/ucxx

sccache --show-adv-stats

rapids-upload-conda-to-s3 cpp
