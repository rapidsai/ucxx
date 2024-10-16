#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/libucxx"

export SKBUILD_CMAKE_ARGS="-DUCXX_ENABLE_RMM=ON"

./ci/build_wheel.sh libucxx "${package_dir}" cpp
