#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

source /opt/conda/etc/profile.d/conda.sh
conda activate base
conda config --set conda_build.root-dir /ucxx/.conda-bld

export GIT_DESCRIBE_NUMBER=${GIT_DESCRIBE_NUMBER:-0}
export PYTHON_VERSION=${PYTHON_VERSION:-3.10}
export NUMPY_VERSION=${NUMPY_VERSION:-1.21}
export RAPIDS_DATE_STRING=${RAPIDS_DATE_STRING:-$(date +%y%m%d)}
export RAPIDS_VERSION=${RAPIDS_VERSION:-23.04}
export RAPIDS_CUDA_VERSION=${RAPIDS_CUDA_VERSION:-11.8}

conda mambabuild --python=${PYTHON_VERSION} --numpy=${NUMPY_VERSION} /ucxx/conda/recipes/ucxx/ 2>&1 | tee ucxx-mamba-build-docker.log
