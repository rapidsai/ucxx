# Copyright (c) 2018-2022, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory
cd python/
PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}
PARALLEL_LEVEL=${PARALLEL_LEVEL} python setup.py build_ext --inplace -j${PARALLEL_LEVEL}
python setup.py install --single-version-externally-managed --record=record.txt
