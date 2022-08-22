/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

#include <ucxx/header.h>

namespace ucxx {

// Header class

Header::Header() : next{false}, nframes{0}
{
  std::fill(isCUDA, isCUDA + HeaderFramesSize, false);
  std::fill(size, size + HeaderFramesSize, 0);
}

Header::Header(bool next, size_t nframes, bool isCUDA, size_t size) : next{next}, nframes{nframes}
{
  std::fill(this->isCUDA, this->isCUDA + nframes, isCUDA);
  std::fill(this->size, this->size + nframes, size);
  if (nframes < HeaderFramesSize) {
    std::fill(this->isCUDA + nframes, this->isCUDA + HeaderFramesSize, false);
    std::fill(this->size + nframes, this->size + HeaderFramesSize, 0);
  }
}

Header::Header(bool next, size_t nframes, int* isCUDA, size_t* size) : next{next}, nframes{nframes}
{
  std::copy(isCUDA, isCUDA + nframes, this->isCUDA);
  std::copy(size, size + nframes, this->size);
  if (nframes < HeaderFramesSize) {
    std::fill(this->isCUDA + nframes, this->isCUDA + HeaderFramesSize, false);
    std::fill(this->size + nframes, this->size + HeaderFramesSize, 0);
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

void Header::print()
{
  std::cout << next << " " << nframes;
  std::cout << " { ";
  std::copy(isCUDA, isCUDA + HeaderFramesSize, std::ostream_iterator<bool>(std::cout, " "));
  std::cout << "} { ";
  std::copy(size, size + HeaderFramesSize, std::ostream_iterator<size_t>(std::cout, " "));
  std::cout << "}";
  std::cout << std::endl;
}

std::vector<Header> Header::buildHeaders(std::vector<size_t>& size, std::vector<int>& isCUDA)
{
  const size_t totalFrames = size.size();

  if (isCUDA.size() != totalFrames)
    throw std::length_error("size and isCUDA must have the same length");

  const size_t totalHeaders = (totalFrames + HeaderFramesSize - 1) / HeaderFramesSize;

  std::vector<Header> headers(totalHeaders);

  for (size_t i = 0; i < totalHeaders; ++i) {
    bool hasNext = totalFrames > (i + 1) * HeaderFramesSize;
    size_t headerFrames =
      hasNext ? HeaderFramesSize : HeaderFramesSize - (HeaderFramesSize * (i + 1) - totalFrames);

    size_t idx = i * HeaderFramesSize;
    headers[i] = Header(hasNext, headerFrames, (int*)&isCUDA[idx], (size_t*)&size[idx]);
  }

  return headers;
}

}  // namespace ucxx
