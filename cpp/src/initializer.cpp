/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <ucxx/initializer.h>
#include <ucxx/log.h>

namespace ucxx {
Initializer::Initializer() { parseLogLevel(); }

Initializer& Initializer::getInstance()
{
  static Initializer instance;
  return instance;
}

}  // namespace ucxx
