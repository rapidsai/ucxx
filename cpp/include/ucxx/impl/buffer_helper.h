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

#include <ucxx/buffer_helper.h>

#if UCXX_ENABLE_RMM
#include <rmm/device_buffer.hpp>
#endif

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

std::string Header::serialize()
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

// PyBuffer class
PyBuffer::PyBuffer(void* ptr, PyBufferDeleter deleter, const bool isCUDA, const size_t size)
  : _ptr{std::unique_ptr<void, PyBufferDeleter>(ptr, deleter)},
    _isCUDA{isCUDA},
    _size{size},
    _isValid{true}
{
}

bool PyBuffer::isValid() { return _isValid; }

size_t PyBuffer::getSize() { return _size; }

bool PyBuffer::isCUDA() { return _isCUDA; }

// PyHostBuffer class
PyHostBuffer::PyHostBuffer(const size_t size)
  : PyBuffer(malloc(size), PyHostBuffer::free, false, size)
{
}

std::unique_ptr<void, PyBufferDeleter> PyHostBuffer::get()
{
  _isValid = false;
  return std::move(_ptr);
}

void* PyHostBuffer::release()
{
  _isValid = false;
  return _ptr.release();
}

void* PyHostBuffer::data() { return _ptr.get(); }

void PyHostBuffer::free(void* ptr) { free(ptr); }

#if UCXX_ENABLE_RMM
// PyRMMBuffer class
PyRMMBuffer::PyRMMBuffer(const size_t size)
  : PyBuffer(new rmm::device_buffer(size, rmm::cuda_stream_default), PyRMMBuffer::free, true, size)
{
}

std::unique_ptr<rmm::device_buffer> PyRMMBuffer::get()
{
  _isValid = false;
  return std::unique_ptr<rmm::device_buffer>((rmm::device_buffer*)_ptr.release());
}

void* PyRMMBuffer::data() { return ((rmm::device_buffer*)_ptr.get())->data(); }

void PyRMMBuffer::free(void* ptr)
{
  rmm::device_buffer* p = (rmm::device_buffer*)ptr;
  delete p;
}
#endif

std::unique_ptr<PyBuffer> allocateBuffer(const bool isCUDA, const size_t size)
{
#if UCXX_ENABLE_RMM
  if (isCUDA)
    return std::make_unique<PyRMMBuffer>(size);
  else
#else
  if (isCUDA)
    throw std::runtime_error("RMM support not enabled, please compile with -DUCXX_ENABLE_RMM=1");
#endif
    return std::make_unique<PyHostBuffer>(size);
}

}  // namespace ucxx
