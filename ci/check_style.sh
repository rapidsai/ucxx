#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n checks
conda activate checks

RAPIDS_VERSION_FILE="$(git rev-parse --show-toplevel)/RAPIDS_VERSION"
RAPIDS_VERSION_MAJOR_MINOR="$(cut -d. -f1,2 < "${RAPIDS_VERSION_FILE}")"

FORMAT_FILE_URL="https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_VERSION_MAJOR_MINOR}/cmake-format-rapids-cmake.json"
export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-formats-rapids-cmake.json
mkdir -p "$(dirname "${RAPIDS_CMAKE_FORMAT_FILE}")"
wget -O ${RAPIDS_CMAKE_FORMAT_FILE} "${FORMAT_FILE_URL}"

# Run pre-commit checks
pre-commit run --all-files --show-diff-on-failure
