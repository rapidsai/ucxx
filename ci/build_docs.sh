#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

UCXX_VERSION_MAJOR_MINOR="$(sed -E -e 's/^([0-9]+)\.([0-9]+)\.([0-9]+).*$/\1.\2/' VERSION)"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n docs
conda activate docs

rapids-print-env

export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libucxx/html"
mv html/* "${RAPIDS_DOCS_DIR}/libucxx/html"
popd

RAPIDS_VERSION_NUMBER="${UCXX_VERSION_MAJOR_MINOR}" rapids-upload-docs
