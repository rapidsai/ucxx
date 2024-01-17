#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

export PROJECT_NAME="ucxx"

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${PROJECT_NAME}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# On arm also need to install CMake because treelite needs to be compiled (no wheels available for arm).
if [[ "$(arch)" == "aarch64" ]]; then
    python -m pip install cmake
fi

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/${PROJECT_NAME}*.whl)[test]

# Run smoke tests for aarch64 pull requests
# if [[ "$(arch)" == "aarch64" && "${RAPIDS_BUILD_TYPE}" == "pull-request" ]]; then
#     python ci/wheel_smoke_test.py
# else
#     python -m pytest ./python/${PROJECT_NAME}/tests -k 'not test_sparse_pca_inputs' -n 4 --ignore=python/cuml/tests/dask && python -m pytest ./python/${PROJECT_NAME}/tests -k 'test_sparse_pca_inputs' && python -m pytest ./python/cuml/tests/dask
# fi

echo "Please add meaningful tests here. This file was copied from CuML and needs to be adapted to UCXX's needs."
exit 1