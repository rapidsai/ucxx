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

# TODO: Upstream this to the image.
mamba install rattler-build -c conda-forge

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rattler-build build \
    # rattler-build uses recipe.yaml, not meta.yaml
    --recipe conda/recipes/ucxx/meta.yaml \
    # rattler-build uses variants.yaml, not conda_build_config.yaml
    --variant-config conda/recipes/ucxx/conda_build_config.yaml \
    # rattler-build does not respect .condarc, so channels and the output dir
    # must be explicitly specified
    -c rapidsai-nightly -c conda-forge \
    --output-dir ${RAPIDS_CONDA_BLD_OUTPUT_DIR}
    # The multi-output cache is currently an experimental feature.
    # (https://prefix-dev.github.io/rattler-build/dev/multiple_output_cache/)
    --experimental
    # By default rattler-build adds a timestamp that defeats sccache caching.
    --no-build-id

rapids-upload-conda-to-s3 cpp
