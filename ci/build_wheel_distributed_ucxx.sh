#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

source ./ci/use_wheels_from_prs.sh

package_dir="python/distributed-ucxx"

./ci/build_wheel.sh distributed-ucxx ${package_dir}
