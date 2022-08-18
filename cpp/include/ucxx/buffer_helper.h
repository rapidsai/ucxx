/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>

#if UCXX_ENABLE_RMM
#include <rmm/device_buffer.hpp>
#endif

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

typedef void (*PyBufferDeleter)(void*);

class PyBuffer {
 protected:
  std::unique_ptr<void, PyBufferDeleter> _ptr{nullptr, [](void*) {}};
  bool _isCUDA{false};
  size_t _size{0};
  bool _isValid{false};

 public:
  PyBuffer()                = delete;
  PyBuffer(const PyBuffer&) = delete;
  PyBuffer& operator=(PyBuffer const&) = delete;
  PyBuffer(PyBuffer&& o)               = delete;
  PyBuffer& operator=(PyBuffer&& o) = delete;

  PyBuffer(void* ptr, PyBufferDeleter deleter, const bool isCUDA, const size_t size);

  bool isValid();

  size_t getSize();

  bool isCUDA();

  virtual void* data() = 0;
};

class PyHostBuffer : public PyBuffer {
 public:
  PyHostBuffer(const size_t size);

  std::unique_ptr<void, PyBufferDeleter> get();

  void* release();

  void* data();

  static void free(void* ptr);
};

#if UCXX_ENABLE_RMM
class PyRMMBuffer : public PyBuffer {
 public:
  PyRMMBuffer(const size_t size);

  std::unique_ptr<rmm::device_buffer> get();

  void* data();

  static void free(void* ptr);
};
#endif

std::unique_ptr<PyBuffer> allocateBuffer(const bool isCUDA, const size_t size);

typedef PyHostBuffer* PyHostBufferPtr;
#if UCXX_ENABLE_RMM
typedef PyRMMBuffer* PyRMMBufferPtr;
#endif

}  // namespace ucxx
