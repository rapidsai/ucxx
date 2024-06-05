#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n docs
conda activate docs

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libucxx

export UCXX_VERSION="$(sed -E -e 's/^([0-9]+)\.([0-9]+)\.([0-9]+).*$/\1.\2.\3/' VERSION)"
export UCXX_VERSION_MAJOR_MINOR="$(sed -E -e 's/^([0-9]+)\.([0-9]+)\.([0-9]+).*$/\1.\2/' VERSION)"
export RAPIDS_VERSION_NUMBER="24.06"
export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libucxx/html"
mv html/* "${RAPIDS_DOCS_DIR}/libucxx/html"
popd

rapids-upload-docs
