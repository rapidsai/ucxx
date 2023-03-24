#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

cmake --install cpp/build --component python
./build.sh ucxx
