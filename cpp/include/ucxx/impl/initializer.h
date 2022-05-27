/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucxx/log.h>

namespace ucxx {
UCXXInitializer::UCXXInitializer() { parseLogLevel(); }

UCXXInitializer& UCXXInitializer::getInstance()
{
  static UCXXInitializer instance;
  return instance;
}

}  // namespace ucxx
