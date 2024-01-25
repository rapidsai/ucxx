#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

PROJECT_NAME="distributed_ucxx"

source "$(dirname "$0")/test_utils.sh"
source "$(dirname "$0")/test_common.sh"

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${PROJECT_NAME}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# Install previously built ucxx wheel
RAPIDS_PY_WHEEL_NAME="ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-ucxx-dep
python -m pip install ./local-ucxx-dep/ucxx*.whl

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/${PROJECT_NAME}*.whl)[test]

# Run smoke tests for aarch64 pull requests
# if [[ "$(arch)" == "aarch64" && "${RAPIDS_BUILD_TYPE}" == "pull-request" ]]; then
#     python ci/wheel_smoke_test.py
# else
#     python -m pytest ./python/${PROJECT_NAME}/tests -k 'not test_sparse_pca_inputs' -n 4 --ignore=python/cuml/tests/dask && python -m pytest ./python/${PROJECT_NAME}/tests -k 'test_sparse_pca_inputs' && python -m pytest ./python/cuml/tests/dask
# fi

print_ucx_config

rapids-logger "Distributed Tests"
# run_distributed_ucxx_tests    PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION   ENABLE_PYTHON_FUTURE
run_distributed_ucxx_tests      thread          1                           1
