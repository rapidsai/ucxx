#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

export ucxx_ROOT="$(realpath ./cpp/build)"
./build.sh -n -v libucxx libucxx_python benchmarks examples tests --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
cmake --install cpp/build --component examples
