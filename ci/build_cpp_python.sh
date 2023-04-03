#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source rapids-env-update

rapids-print-env

rapids-logger "Begin py build"

rapids-mamba-retry mambabuild \
  conda/recipes/ucxx

rapids-upload-conda-to-s3 python
