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

#rapids-conda-retry mambabuild \
#  conda/recipes/ucxx

#mamba install rattler-build -c conda-forge

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rattler-build build --recipe conda/recipes/ucxx/meta.yaml --variant-config conda/recipes/ucxx/conda_build_config.yaml -c rapidsai-nightly -c conda-forge --output-dir /tmp/output/ --experimental --no-build-id

rapids-upload-conda-to-s3 cpp
