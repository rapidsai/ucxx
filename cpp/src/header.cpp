/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

#include <ucxx/header.h>

namespace ucxx {

Header::Header(bool next, size_t nframes, int* isCUDA, size_t* size) : next{next}, nframes{nframes}
{
  std::copy(isCUDA, isCUDA + nframes, this->isCUDA.begin());
  std::copy(size, size + nframes, this->size.begin());
  if (nframes < HeaderFramesSize) {
    std::fill(this->isCUDA.begin() + nframes, this->isCUDA.begin() + HeaderFramesSize, false);
    std::fill(this->size.begin() + nframes, this->size.begin() + HeaderFramesSize, 0);
  }
}

Header::Header(std::string serializedHeader) { deserialize(serializedHeader); }

size_t Header::dataSize() { return sizeof(next) + sizeof(nframes) + sizeof(isCUDA) + sizeof(size); }

const std::string Header::serialize() const
{
  std::stringstream ss;

  ss.write((char const*)&next, sizeof(next));
  ss.write((char const*)&nframes, sizeof(nframes));
  for (size_t i = 0; i < HeaderFramesSize; ++i)
    ss.write((char const*)&isCUDA[i], sizeof(isCUDA[i]));
  for (size_t i = 0; i < HeaderFramesSize; ++i)
    ss.write((char const*)&size[i], sizeof(size[i]));

  return ss.str();
}

void Header::deserialize(const std::string& serializedHeader)
{
  std::stringstream ss{serializedHeader};

  ss.read((char*)&next, sizeof(next));
  ss.read((char*)&nframes, sizeof(nframes));
  for (size_t i = 0; i < HeaderFramesSize; ++i)
    ss.read((char*)&isCUDA[i], sizeof(isCUDA[i]));
  for (size_t i = 0; i < HeaderFramesSize; ++i)
    ss.read((char*)&size[i], sizeof(size[i]));
}

std::vector<Header> Header::buildHeaders(std::vector<size_t>& size, std::vector<int>& isCUDA)
{
  const size_t totalFrames = size.size();

  if (isCUDA.size() != totalFrames)
    throw std::length_error("size and isCUDA must have the same length");

  const size_t totalHeaders = (totalFrames + HeaderFramesSize - 1) / HeaderFramesSize;

  std::vector<Header> headers;

  for (size_t i = 0; i < totalHeaders; ++i) {
    bool hasNext = totalFrames > (i + 1) * HeaderFramesSize;
    size_t headerFrames =
      hasNext ? HeaderFramesSize : HeaderFramesSize - (HeaderFramesSize * (i + 1) - totalFrames);

    size_t idx = i * HeaderFramesSize;
    headers.push_back(Header(hasNext, headerFrames, (int*)&isCUDA[idx], (size_t*)&size[idx]));
  }

  return headers;
}

}  // namespace ucxx
