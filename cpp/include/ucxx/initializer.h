/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

namespace ucxx {

class Initializer {
 private:
  Initializer();

  Initializer(const Initializer&) = delete;
  Initializer& operator=(Initializer const&) = delete;
  Initializer(Initializer&& o)               = delete;
  Initializer& operator=(Initializer&& o) = delete;

 public:
  static Initializer& getInstance();
};

}  // namespace ucxx
