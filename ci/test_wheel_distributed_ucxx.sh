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

# TODO: We need distributed installed in developer mode to provide test utils,
# we still need to match to the `rapids-dask-dependency` version.
rapids-logger "Install Distributed in developer mode"
git clone https://github.com/dask/distributed /tmp/distributed
python -m pip install -e /tmp/distributed

# Run smoke tests for aarch64 pull requests
if [[ "$(arch)" == "aarch64" && "${RAPIDS_BUILD_TYPE}" == "pull-request" ]]; then
  rapids-logger "Distributed Smoke Tests"
  python -m pytest -vs ci/wheel_smoke_test.py
else
  rapids-logger "Distributed Tests"

  # run_distributed_ucxx_tests    PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION   ENABLE_PYTHON_FUTURE
  run_distributed_ucxx_tests      thread          1                           1
fi
