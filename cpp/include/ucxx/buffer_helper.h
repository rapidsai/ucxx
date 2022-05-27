/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>

#include <rmm/device_buffer.hpp>

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

  std::string serialize();

  void deserialize(const std::string& serializedHeader);

  void print();
};

typedef void (*UCXXPyBufferDeleter)(void*);

class UCXXPyBuffer {
 protected:
  std::unique_ptr<void, UCXXPyBufferDeleter> _ptr{nullptr, [](void*) {}};
  bool _isCUDA{false};
  size_t _size{0};
  bool _isValid{false};

 public:
  UCXXPyBuffer()                    = delete;
  UCXXPyBuffer(const UCXXPyBuffer&) = delete;
  UCXXPyBuffer& operator=(UCXXPyBuffer const&) = delete;
  UCXXPyBuffer(UCXXPyBuffer&& o)               = delete;
  UCXXPyBuffer& operator=(UCXXPyBuffer&& o) = delete;

  UCXXPyBuffer(void* ptr, UCXXPyBufferDeleter deleter, const bool isCUDA, const size_t size);

  bool isValid();

  size_t getSize();

  bool isCUDA();

  virtual void* data() = 0;
};

class UCXXPyHostBuffer : public UCXXPyBuffer {
 public:
  UCXXPyHostBuffer(const size_t size);

  std::unique_ptr<void, UCXXPyBufferDeleter> get();

  void* release();

  void* data();

  static void free(void* ptr);
};

class UCXXPyRMMBuffer : public UCXXPyBuffer {
 public:
  UCXXPyRMMBuffer(const size_t size);

  std::unique_ptr<rmm::device_buffer> get();

  void* data();

  static void free(void* ptr);
};

std::unique_ptr<UCXXPyBuffer> allocateBuffer(const bool isCUDA, const size_t size);

typedef UCXXPyHostBuffer* UCXXPyHostBufferPtr;
typedef UCXXPyRMMBuffer* UCXXPyRMMBufferPtr;

}  // namespace ucxx
