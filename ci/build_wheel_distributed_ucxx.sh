#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/distributed-ucxx"

./ci/build_wheel.sh distributed-ucxx ${package_dir}
