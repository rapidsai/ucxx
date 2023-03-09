#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

export ucxx_ROOT="$(realpath ./cpp/build)"
./build.sh -n -v libucxx libucxx_python benchmarks tests --build_metrics --incl_cache_stats --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
