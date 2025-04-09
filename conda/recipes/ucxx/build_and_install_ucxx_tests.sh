#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

cmake --install cpp/python/build
./build.sh ucxx ucxx_tests --cmake-args="\"-DFIND_UCXX_PYTHON=ON -Ducxx-python_DIR=$(realpath ./cpp/python/build)\""
