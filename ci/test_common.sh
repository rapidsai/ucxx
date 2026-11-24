#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail


log_command() {
  CMD_LINE=$1
  echo -e "\e[1mRunning: \n ${CMD_LINE}\e[0m"
}

log_message() {
  MESSAGE=$1
  color_format="\e[1;32m"
  disable_format="\e[0m"
  echo -e "${color_format}${MESSAGE}${disable_format}"
}

print_system_stats() {
  rapids-logger "Number of CPU cores"
  nproc --all

  rapids-logger "Check GPU usage"
  nvidia-smi

  rapids-logger "Check NICs"
  awk 'END{print $1}' /etc/hosts
  cat /etc/hosts

  rapids-logger "Check maximum number of pending connections (somaxconn)"
  cat /proc/sys/net/core/somaxconn

  print_node_health
}

print_node_health() {
  # Report-only: helps attributing failures to degraded CI nodes, never fails the job.
  rapids-logger "Node health"
  python "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/node_health.py" || true
}

print_ucx_config() {
  rapids-logger "UCX Version and Build Configuration"

  ucx_info -v
}

configure_ucx_tls() {
  local ucx_version

  if command -v ucx_info > /dev/null; then
    ucx_version="$(ucx_info -v | sed -nE 's/^# Library version: ([0-9]+\.[0-9]+\.[0-9]+).*/\1/p' | head -n 1)"
  else
    # ucx_info is not always available (wheels do not ship executables). Rely on
    # Python to determine UCX version as a fallback.
    ucx_version="$(python -c 'import ucxx; print(ucxx.__ucx_version__)')"
  fi

  # See https://github.com/rapidsai/ucxx/issues/733
  # TODO: Remove when UCX 1.23 or higher becomes minimum supported.
  if [[ "${ucx_version}" == 1.21.* ]]; then
    export UCX_TLS=^cuda_ipc
  fi
}

configure_crash_backtraces() {
  # Print native (C/C++) backtraces when a process that loaded UCX (any process
  # importing ucxx, including pytest-xdist workers and their subprocesses)
  # receives a fatal signal (SIGSEGV, SIGBUS, SIGILL, SIGFPE). UCX's signal handler
  # chains to previously installed handlers (e.g., Python's `faulthandler`), so
  # Python stacks are still printed afterwards. Nothing is printed on success.
  #
  # - When GDB is available and allowed to attach to its parent process, use
  #   `UCX_HANDLE_ERRORS=debug`: UCX forks GDB which prints the stacks of all
  #   threads (see `crash_backtrace.sh`), bounded by a timeout.
  # - Otherwise use `UCX_HANDLE_ERRORS=bt`: UCX prints the stack of the crashing
  #   thread itself.
  #
  # In CI the former applies: GDB ships with `rapidsai/citestwheel` images and is
  # installed in conda test environments (`dependencies.yaml`), and test jobs run as
  # root with `--cap-add CAP_SYS_PTRACE`. The latter covers other environments,
  # e.g. local runs or devcontainers without GDB or ptrace permissions.
  local ci_dir cap_eff ptrace_scope

  if [[ -n "${UCX_HANDLE_ERRORS:-}" ]]; then
    echo "UCX_HANDLE_ERRORS=${UCX_HANDLE_ERRORS} already set, not overriding"
    return
  fi

  export UCX_HANDLE_ERRORS=bt

  if command -v gdb > /dev/null; then
    # Attaching to the parent process requires either Yama's `ptrace_scope=0` or
    # CAP_SYS_PTRACE (bit 19 of the effective capabilities) with `ptrace_scope<3`.
    ptrace_scope="$(cat /proc/sys/kernel/yama/ptrace_scope 2> /dev/null || echo 0)"
    cap_eff="$(awk '/^CapEff:/ {print $2}' /proc/self/status 2> /dev/null || echo 0)"
    if [[ "${ptrace_scope}" -eq 0 ]] || \
      { [[ "${ptrace_scope}" -lt 3 ]] && (( (16#${cap_eff:-0} >> 19) & 1 )); }; then
      ci_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
      export UCX_HANDLE_ERRORS=debug
      export UCX_GDB_COMMAND="${ci_dir}/crash_backtrace.sh"
    fi
  fi

  echo "Native backtraces on crash: UCX_HANDLE_ERRORS=${UCX_HANDLE_ERRORS} UCX_GDB_COMMAND=${UCX_GDB_COMMAND:-}"
}

run_port_retry() {
  MAX_ATTEMPTS=${1}
  RUN_TYPE=${2}
  PROGRESS_MODE=${3}
  RUN_FUNCTION=${4}

  set +e
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    echo "Attempt ${attempt}/${MAX_ATTEMPTS} to run ${RUN_TYPE}"

    _SERVER_PORT=$((_SERVER_PORT + 1))    # Use different ports every time to prevent `Device is busy`

    # Call the provided function with the required arguments
    ${RUN_FUNCTION} ${_SERVER_PORT} "${PROGRESS_MODE}"

    LAST_STATUS=$?
    if [ ${LAST_STATUS} -eq 0 ]; then
      break;
    fi
    sleep 1
  done
  set -e

  if [ "${LAST_STATUS}" -ne 0 ]; then
    echo "Failure running ${RUN_TYPE} after ${MAX_ATTEMPTS} attempts"
    exit "$LAST_STATUS"
  fi
}
