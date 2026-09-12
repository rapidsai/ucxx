#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

source "$(dirname "$0")/test_common.sh"

configure_ucx_tls

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

workers="${UCXX_TAG_MULTI_DIAGNOSTIC_WORKERS:?Must specify 1 or 4 diagnostic workers}"
if [[ "${workers}" != 1 && "${workers}" != 4 ]]; then
  echo "UCXX_TAG_MULTI_DIAGNOSTIC_WORKERS must be 1 or 4" >&2
  exit 2
fi

tests=(
  "python/ucxx/ucxx/_lib_async/tests/test_send_recv_multi.py::test_send_recv_cupy[|u1-3-1048576]"
  "python/ucxx/ucxx/_lib_async/tests/test_send_recv_multi.py::test_send_recv_cupy[|u1-4-1048576]"
  "python/ucxx/ucxx/_lib_async/tests/test_send_recv_multi.py::test_send_recv_cupy[<i8-2-65536]"
  "python/ucxx/ucxx/_lib_async/tests/test_send_recv_multi.py::test_send_recv_cupy[<i8-3-65536]"
  "python/ucxx/ucxx/_lib_async/tests/test_send_recv_multi.py::test_send_recv_numba[<i8-4-65536]"
  "python/ucxx/ucxx/_lib_async/tests/test_send_recv_multi.py::test_send_recv_numba[f8-8-65536]"
)

for iteration in $(seq 1 100); do
  log_message "Tag-multi diagnostic iteration ${iteration}/100 with ${workers} workers"
  UCXPY_PROGRESS_MODE=thread \
  UCXPY_ENABLE_DELAYED_SUBMISSION=0 \
  UCXPY_ENABLE_PYTHON_FUTURE=0 \
  UCXX_TAG_MULTI_DIAGNOSTICS=1 \
  python "${TIMEOUT_TOOL_PATH}" --enable-python $((5*60)) \
    python -m pytest -n "${workers}" --force-reruns 0 --import-mode=append -vs "${tests[@]}"
done
