# Copyright (c) 2018-2022, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory
cd python/
PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}
$PYTHON -m pip install -vv --no-deps --ignore-installed . --global-option="build_ext" --global-option="-j${PARALLEL_LEVEL}"
