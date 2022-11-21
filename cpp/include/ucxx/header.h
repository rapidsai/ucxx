/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <string>
#include <vector>

namespace ucxx {

const size_t HeaderFramesSize = 100;

class Header {
 public:
  bool next;
  size_t nframes;
  bool isCUDA[HeaderFramesSize];
  size_t size[HeaderFramesSize];

  Header();

  Header(bool next, size_t nframes, bool isCUDA, size_t size);

  Header(bool next, size_t nframes, int* isCUDA, size_t* size);

  Header(std::string serializedHeader);

  static size_t dataSize();

  const std::string serialize() const;

  void deserialize(const std::string& serializedHeader);

  void print();

  static std::vector<Header> buildHeaders(std::vector<size_t>& size, std::vector<int>& isCUDA);
};

}  // namespace ucxx
