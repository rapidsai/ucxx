#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
LIBRMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact rmm 2184 cpp wheel
)
RMM_WHEELHOUSE=$(
  rapids-get-pr-artifact rmm 2184 python wheel --stable
)

cat > "${PIP_CONSTRAINT}" <<EOF
librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRMM_WHEELHOUSE}"/librmm_*.whl)
rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RMM_WHEELHOUSE}"/rmm_*.whl)
EOF
