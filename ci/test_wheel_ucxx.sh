#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source rapids-init-pip

package_name="ucxx"

source "$(dirname "$0")/test_common.sh"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the packages built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBUCXX_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libucxx ucxx --cuda "$RAPIDS_CUDA_VERSION")")
UCXX_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python ucxx ucxx --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

python -m venv libucxx-env
. libucxx-env/bin/activate

rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${LIBUCXX_WHEELHOUSE}"/libucxx_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"
python -c "import libucxx; assert libucxx.load_library() is not None"
deactivate

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "${LIBUCXX_WHEELHOUSE}/libucxx_${RAPIDS_PY_CUDA_SUFFIX}"*.whl \
    "$(echo "${UCXX_WHEELHOUSE}"/"${package_name}_${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

print_system_stats

# TEMPORARY debugging of the hang seen in nightly CI on V100. Nothing below fails the script early;
# every step runs and the job's exit status reflects all of them.
#
# 1. Standalone CUDA driver API reproducer (no UCX/ucxx), one short run (it was clean in 10x60s
#    runs already, so this is just for the record). Exit status 3 means it could not be built.
rapids-logger "Run standalone cuMemRetainAllocationHandle reproducer (short)"
repro_status=0
N_REPRO_ATTEMPTS=1 REPRO_DURATION=20 ./ci/debug/build_and_run_retain_repro.sh || repro_status=$?

# 2. A/B of the receive-buffer allocator used by ucxx (UCXX_DEBUG_CCCL_BUFFER_ALLOCATOR, see
#    cpp/src/buffer_cccl.cpp) on this very runner with the same wheels:
#      pool        = CCCL's stream-ordered default device memory pool (the shipping behavior)
#      cudamalloc  = plain cudaMalloc/cudaFree
#    Each case is a loop that stops at its first failure. Case 1 should hang if this runner is
#    "bad"; if so, it also confirms the runner is a valid test bed for the other cases.
declare -a case_names=()
declare -a case_status=()

run_case() {
  local name="$1" allocator="$2" log_level="$3" attempts="$4"
  rapids-logger "CASE ${name}: allocator=${allocator}, UCX_LOG_LEVEL=${log_level}, up to ${attempts} attempts"
  local status=0
  UCXX_DEBUG_CCCL_BUFFER_ALLOCATOR="${allocator}" \
  BENCH_UCX_LOG_LEVEL="${log_level}" \
  N_ATTEMPTS="${attempts}" \
  DISABLE_CYTHON=1 ./ci/debug_benchmark_loop.sh || status=$?
  case_names+=("${name}")
  case_status+=("${status}")
}

run_case "pool-trace"        pool       trace 3
run_case "cudamalloc-trace"  cudamalloc trace 10
run_case "pool-info"         pool       info  5
run_case "cudamalloc-info"   cudamalloc info  10

rapids-logger "SUMMARY (0 = all attempts passed, 124 = timeout/hang, other = failure)"
echo "standalone reproducer: status=${repro_status} (0=clean, 3=skipped)"
overall=0
for i in "${!case_names[@]}"; do
  printf '%-20s status=%s\n' "${case_names[$i]}" "${case_status[$i]}"
  if [ "${case_status[$i]}" -ne 0 ] && [ "${overall}" -eq 0 ]; then overall="${case_status[$i]}"; fi
done
if [ "${repro_status}" -ne 0 ] && [ "${repro_status}" -ne 3 ] && [ "${overall}" -eq 0 ]; then
  overall="${repro_status}"
fi
exit "${overall}"
