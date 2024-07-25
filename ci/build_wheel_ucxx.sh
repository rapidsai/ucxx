#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python"

./ci/build_wheel.sh ucxx ${package_dir}
