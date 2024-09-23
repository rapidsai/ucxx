#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/libucxx"

./ci/build_wheel.sh libucxx ${package_dir}
