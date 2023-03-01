/**
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>

namespace ucxx {

namespace utils {

/*
 * @brief Check whether Python support is available.
 *
 * Check that binary was built with Python support and `libucxx_python.so` is in the
 * library path. The check is done by attempting to `dlopen` the library, returning whether
 * both conditions are met.
 *
 * @returns whether Python support is available.
 */
bool isPythonAvailable();

}  // namespace utils

}  // namespace ucxx
