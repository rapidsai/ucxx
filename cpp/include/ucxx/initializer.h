/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

namespace ucxx {

class UCXXInitializer {
 private:
  UCXXInitializer();

  UCXXInitializer(const UCXXInitializer&) = delete;
  UCXXInitializer& operator=(UCXXInitializer const&) = delete;
  UCXXInitializer(UCXXInitializer&& o)               = delete;
  UCXXInitializer& operator=(UCXXInitializer&& o) = delete;

 public:
  static UCXXInitializer& getInstance();
};

}  // namespace ucxx
