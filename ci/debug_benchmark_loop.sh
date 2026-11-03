#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# TEMPORARY DEBUGGING SCRIPT (do not merge).
#
# Repeatedly runs the single benchmark that hangs in nightly CI on V100 / CUDA 12.2.2:
#
#   ucxx-async -o cupy --reuse-alloc --n-buffers 8 --progress-mode thread,
#   delayed submission 0, python future 0
#
# with UCX_LOG_LEVEL=trace. It stops on the first failure (non-zero exit status or timeout) and
# prints the head and tail of the UCX trace log for that attempt, along with the stack traces
# captured by `timeout_with_stack.py`. Trace logs of successful attempts are discarded.
#
# Tunables (environment variables):
#   N_ATTEMPTS           Maximum number of benchmark runs (default: 25).
#   BENCH_UCX_LOG_LEVEL  Value of UCX_LOG_LEVEL for the benchmark (default: trace).
#   BENCH_TIMEOUT        Per-attempt timeout in seconds (default: 120, same as run_python.sh).
#   LOG_HEAD_LINES       Lines from the start of the log to print on failure (default: 3000).
#   LOG_TAIL_LINES       Lines from the end of the log to print on failure (default: 30000).
#   LOG_MAX_KIB          Per-attempt log size cap in KiB (default: 4194304, i.e. 4 GiB).

set -uo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
TIMEOUT_TOOL_PATH="${SCRIPT_DIR}/timeout_with_stack.py"

source "${SCRIPT_DIR}/test_common.sh"
# test_common.sh enables `set -e`; failures are handled explicitly below.
set +e

configure_ucx_tls

N_ATTEMPTS="${N_ATTEMPTS:-25}"
BENCH_UCX_LOG_LEVEL="${BENCH_UCX_LOG_LEVEL:-trace}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-120}"
LOG_HEAD_LINES="${LOG_HEAD_LINES:-3000}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-30000}"
LOG_MAX_KIB="${LOG_MAX_KIB:-4194304}"

LOG_DIR="$(mktemp -d)"
export LOG_DIR LOG_MAX_KIB BENCH_UCX_LOG_LEVEL

log_message "Environment"
env | sort | grep -E '^(UCX|UCXPY|CUDA|LD_LIBRARY_PATH|RAPIDS)' || true
nvidia-smi || true
python -c "import ucxx; print('ucxx', ucxx.__version__, 'UCX', ucxx.__ucx_version__)" || true
python -m pip list 2>/dev/null | grep -i -E 'ucx|cupy|cuda|numpy|rmm|libucx' || true

# Run by the timeout wrapper. The benchmark's (very verbose) stdout/stderr go to a log file, while
# the wrapper's own output (including the stack traces of a hung process) stays on the console.
cat > "${LOG_DIR}/run_benchmark.sh" <<'EOF'
#!/bin/bash
ulimit -f "${LOG_MAX_KIB}"
exec python -m ucxx.benchmarks.send_recv --backend ucxx-async \
  -o cupy --reuse-alloc -n 8MiB --n-buffers 8 --progress-mode thread \
  > "${LOG_DIR}/benchmark.log" 2>&1
EOF
chmod +x "${LOG_DIR}/run_benchmark.sh"

log_message "Debugging benchmark loop: up to ${N_ATTEMPTS} attempts, UCX_LOG_LEVEL=${BENCH_UCX_LOG_LEVEL}"

for attempt in $(seq 1 "${N_ATTEMPTS}"); do
  rm -f "${LOG_DIR}/benchmark.log"
  echo "=== Attempt ${attempt}/${N_ATTEMPTS} ($(date -u +%H:%M:%S))"

  start=$(date +%s)
  # Same settings as the corresponding run_py_benchmark call in run_python.sh, plus trace logging.
  UCX_KEEPALIVE_INTERVAL=1ms \
  UCXPY_ENABLE_DELAYED_SUBMISSION=0 \
  UCXPY_ENABLE_PYTHON_FUTURE=0 \
  UCX_LOG_LEVEL="${BENCH_UCX_LOG_LEVEL}" \
  python "${TIMEOUT_TOOL_PATH}" --enable-python "${BENCH_TIMEOUT}" \
    "${LOG_DIR}/run_benchmark.sh"
  status=$?
  elapsed=$(( $(date +%s) - start ))
  log_size=$(du -h "${LOG_DIR}/benchmark.log" 2>/dev/null | cut -f1)

  if [ "${status}" -eq 0 ]; then
    echo "Attempt ${attempt} passed in ${elapsed}s (trace log: ${log_size:-unknown}, discarded)"
    continue
  fi

  log_message "FAILURE on attempt ${attempt}/${N_ATTEMPTS}: exit status ${status} after ${elapsed}s (trace log: ${log_size:-unknown})"
  echo "::group::benchmark log: first ${LOG_HEAD_LINES} lines"
  head -n "${LOG_HEAD_LINES}" "${LOG_DIR}/benchmark.log"
  echo "::endgroup::"
  echo "::group::benchmark log: last ${LOG_TAIL_LINES} lines"
  tail -n "${LOG_TAIL_LINES}" "${LOG_DIR}/benchmark.log"
  echo "::endgroup::"
  echo "::group::benchmark output, tracebacks and UCX/UCXX warnings and errors from the log"
  grep -E -v '^\[[0-9]+\.[0-9]+\] ' "${LOG_DIR}/benchmark.log" | head -n 300
  grep -E '^\[[0-9]+\.[0-9]+\] .* UCX[A-Z]* +(ERROR|WARN) ' "${LOG_DIR}/benchmark.log" | head -n 300
  echo "::endgroup::"
  nvidia-smi || true
  echo "Failure running Python benchmark on attempt ${attempt}/${N_ATTEMPTS}"
  exit "${status}"
done

log_message "All ${N_ATTEMPTS} attempts passed; hang not reproduced"
