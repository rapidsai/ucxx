#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

cmake --install cpp/build
cmake --install cpp/build --component benchmarks

# For some reason RAPIDS headers are getting installed causing clobbering, which shouldn't happen.
# To workaround this issue for now, just remove all the RAPIDS headers that were installed to avoid clobbering.
# xref: https://github.com/rapidsai/ucxx/issues/20
rm -rf "${PREFIX}/include/rapids"
