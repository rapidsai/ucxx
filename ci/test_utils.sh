#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


log_command() {
  CMD_LINE=$1
  echo -e "\e[1mRunning: \n ${CMD_LINE}\e[0m"
}

print_system_stats() {
  rapids-logger "Check GPU usage"
  nvidia-smi

  rapids-logger "Check NICs"
  awk 'END{print $1}' /etc/hosts
  cat /etc/hosts
}

print_ucx_config() {
  rapids-logger "UCX Version and Build Configuration"
  if [ $(which ucx_info) == "" ]; then
    echo "ucx_info not found"
  else
    ucx_info -v
  fi
}
