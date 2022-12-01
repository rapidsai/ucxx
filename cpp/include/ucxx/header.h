/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <array>
#include <string>
#include <vector>

namespace ucxx {

const size_t HeaderFramesSize = 100;

class Header {
 private:
  void deserialize(const std::string& serializedHeader);

 public:
  bool next;
  size_t nframes;
  std::array<int, HeaderFramesSize> isCUDA;
  std::array<size_t, HeaderFramesSize> size;

  Header() = delete;

  Header(bool next, size_t nframes, int* isCUDA, size_t* size);

  Header(std::string serializedHeader);

  static size_t dataSize();

  const std::string serialize() const;

  static std::vector<Header> buildHeaders(std::vector<size_t>& size, std::vector<int>& isCUDA);
};

}  // namespace ucxx
