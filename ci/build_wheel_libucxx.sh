#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

source ./ci/use_wheels_from_prs.sh

package_dir="python/libucxx"

./ci/build_wheel.sh libucxx ${package_dir}
