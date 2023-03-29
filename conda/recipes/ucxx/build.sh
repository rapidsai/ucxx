#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

export ucxx_ROOT="$(realpath ./cpp/build)"
./build.sh -n -v libucxx libucxx_python benchmarks examples tests --build_metrics --incl_cache_stats --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
