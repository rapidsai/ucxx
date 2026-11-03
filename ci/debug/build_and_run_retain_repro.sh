#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# TEMPORARY DEBUGGING SCRIPT (do not merge).
#
# Builds ci/debug/retain_handle_repro.cpp (driver API only: needs cuda.h and libcuda.so.1) and
# runs it repeatedly, stopping at the first failure. See the header of the .cpp file for what it
# does and, just as important, what a pass or a failure does and does not prove.
#
# Exit status: 0 = every run was clean, 3 = could not build (nothing was run), otherwise the
# status of the first failing run (124 = hang detected by the program's watchdog or by the
# timeout wrapper).
#
# Tunables (environment variables):
#   N_REPRO_ATTEMPTS   Number of runs (default: 10).
#   REPRO_DURATION     Seconds per run (default: 60).
#   REPRO_STUCK_SEC    A driver call in flight for longer than this is a hang (default: 20).
#   REPRO_ALLOC        pool | vmm | mixed (default: mixed).
#   REPRO_RETAINERS    Number of querying threads (default: 2).
#   REPRO_EXTRA_ARGS   Extra arguments for the program (e.g. "--slot-bytes 1048576").

set -uo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
TIMEOUT_TOOL_PATH="${SCRIPT_DIR}/../timeout_with_stack.py"
SRC="${SCRIPT_DIR}/retain_handle_repro.cpp"

N_REPRO_ATTEMPTS="${N_REPRO_ATTEMPTS:-10}"
REPRO_DURATION="${REPRO_DURATION:-60}"
REPRO_STUCK_SEC="${REPRO_STUCK_SEC:-20}"
REPRO_ALLOC="${REPRO_ALLOC:-mixed}"
REPRO_RETAINERS="${REPRO_RETAINERS:-2}"
read -r -a REPRO_EXTRA <<< "${REPRO_EXTRA_ARGS:-}"

msg() { echo -e "\e[1;32m$1\e[0m"; }
skip() { echo -e "\e[1;33mSKIPPING standalone reproducer: $1\e[0m"; exit 3; }

msg "Standalone cuMemRetainAllocationHandle reproducer: build"

# --- compiler --------------------------------------------------------------------------------
CXX="$(command -v g++ || command -v c++ || true)"
if [ -z "${CXX}" ]; then
  echo "No C++ compiler found; trying to install g++ with apt-get"
  if command -v apt-get > /dev/null; then
    { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends g++; } \
      > /tmp/apt_install_gxx.log 2>&1 || tail -n 20 /tmp/apt_install_gxx.log
    CXX="$(command -v g++ || true)"
  fi
fi
[ -n "${CXX}" ] || skip "no C++ compiler available"

# --- cuda.h ----------------------------------------------------------------------------------
CUDA_INCLUDE=""
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2> /dev/null || true)"
for d in \
    ${SITE_PACKAGES:+"${SITE_PACKAGES}"/nvidia/cuda_runtime/include} \
    ${SITE_PACKAGES:+"${SITE_PACKAGES}"/nvidia/cu12/include} \
    ${SITE_PACKAGES:+"${SITE_PACKAGES}"/nvidia/cu13/include} \
    "${CUDA_HOME:-/nonexistent}/include" \
    /usr/local/cuda/include; do
  if [ -f "${d}/cuda.h" ]; then CUDA_INCLUDE="${d}"; break; fi
done
if [ -z "${CUDA_INCLUDE}" ] && [ -n "${SITE_PACKAGES}" ]; then
  CUDA_H="$(find "${SITE_PACKAGES}/nvidia" -name cuda.h -path '*include*' 2> /dev/null | head -n 1)"
  [ -n "${CUDA_H}" ] && CUDA_INCLUDE="$(dirname "${CUDA_H}")"
fi
[ -n "${CUDA_INCLUDE}" ] || skip "cuda.h not found"

# --- libcuda.so.1 ----------------------------------------------------------------------------
LIBCUDA="$(ldconfig -p 2> /dev/null | awk '/libcuda\.so\.1 / && /x86-64|aarch64/ {print $NF; exit}')"
if [ -z "${LIBCUDA}" ]; then
  for d in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/local/nvidia/lib64 /usr/local/cuda/compat; do
    if [ -e "${d}/libcuda.so.1" ]; then LIBCUDA="${d}/libcuda.so.1"; break; fi
  done
fi
[ -n "${LIBCUDA}" ] || skip "libcuda.so.1 not found"

BUILD_DIR="$(mktemp -d)"
BIN="${BUILD_DIR}/retain_handle_repro"
echo "compiler: ${CXX} ($(${CXX} --version | head -n 1))"
echo "cuda.h:   ${CUDA_INCLUDE}/cuda.h"
echo "libcuda:  ${LIBCUDA}"
if ! "${CXX}" -O2 -std=c++17 -Wall -pthread "${SRC}" -I"${CUDA_INCLUDE}" \
      -L"$(dirname "${LIBCUDA}")" -l:libcuda.so.1 -o "${BIN}"; then
  skip "build failed"
fi

# --- run -------------------------------------------------------------------------------------
nvidia-smi --query-gpu=name,driver_version --format=csv || true

msg "Standalone reproducer: up to ${N_REPRO_ATTEMPTS} runs of ${REPRO_DURATION}s (alloc=${REPRO_ALLOC}, ${REPRO_RETAINERS} retainers)"
for attempt in $(seq 1 "${N_REPRO_ATTEMPTS}"); do
  echo "=== Reproducer attempt ${attempt}/${N_REPRO_ATTEMPTS} ($(date -u +%H:%M:%S))"
  # If the program's watchdog detects a hang it reports which call is stuck and then lingers, so
  # that the wrapper's timeout fires and dumps native stacks of the stuck threads.
  python "${TIMEOUT_TOOL_PATH}" "$((REPRO_DURATION + REPRO_STUCK_SEC + 30))" \
    "${BIN}" --duration "${REPRO_DURATION}" --stuck-sec "${REPRO_STUCK_SEC}" \
    --alloc "${REPRO_ALLOC}" --retainers "${REPRO_RETAINERS}" --linger-sec "$((REPRO_DURATION + REPRO_STUCK_SEC + 120))" \
    ${REPRO_EXTRA[@]+"${REPRO_EXTRA[@]}"}
  status=$?
  if [ "${status}" -ne 0 ]; then
    msg "REPRODUCER FAILURE on attempt ${attempt}/${N_REPRO_ATTEMPTS}: exit status ${status}"
    exit "${status}"
  fi
done

msg "Standalone reproducer: all ${N_REPRO_ATTEMPTS} runs were clean (this proves nothing about the benchmark)"
