#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

export LIB_BUILD_DIR="${PREFIX}/tmp/ucx_build/" 
./build.sh -n -v libucxx libucxx_python benchmarks examples tests --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"

cmake --install ${LIB_BUILD_DIR} --prefix ${PREFIX}/tmp/install/libucxx/
for component in python benchmarks examples testing; do
  cmake --install ${LIB_BUILD_DIR} --component ${component} --prefix ${PREFIX}/tmp/install/${component}/
done
