/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <cstdio>

namespace ucxx {

namespace utils {

FILE* createTextFileDescriptor();

std::string decodeTextFileDescriptor(FILE* textFileDescriptor);

}  // namespace utils

}  // namespace ucxx
