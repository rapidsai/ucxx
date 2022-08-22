/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>
#include <utility>

#include <ucxx/log.h>

#if UCXX_ENABLE_RMM
#include <rmm/device_buffer.hpp>
#endif

namespace ucxx {

enum class BufferType {
  Host = 0,
  RMM,
  Invalid,
};

class Buffer {
 protected:
  BufferType _bufferType{BufferType::Invalid};
  size_t _size;

  Buffer(const BufferType bufferType, const size_t size);

 public:
  Buffer()              = delete;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(Buffer const&) = delete;
  Buffer(Buffer&& o)               = delete;
  Buffer& operator=(Buffer&& o) = delete;

  virtual ~Buffer();

  BufferType getType() const noexcept;

  size_t getSize() const noexcept;

  virtual void* data() = 0;
};

class HostBuffer : public Buffer {
 private:
  void* _buffer;

 public:
  HostBuffer()                  = delete;
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(HostBuffer const&) = delete;
  HostBuffer(HostBuffer&& o)               = delete;
  HostBuffer& operator=(HostBuffer&& o) = delete;

  HostBuffer(const size_t size);

  ~HostBuffer();

  void* release();

  virtual void* data();
};

#if UCXX_ENABLE_RMM
class RMMBuffer : public Buffer {
 private:
  std::unique_ptr<rmm::device_buffer> _buffer;

 public:
  RMMBuffer()                 = delete;
  RMMBuffer(const RMMBuffer&) = delete;
  RMMBuffer& operator=(RMMBuffer const&) = delete;
  RMMBuffer(RMMBuffer&& o)               = delete;
  RMMBuffer& operator=(RMMBuffer&& o) = delete;

  RMMBuffer(const size_t size);

  std::unique_ptr<rmm::device_buffer> release();

  virtual void* data();
};
#endif

Buffer* allocateBuffer(BufferType bufferType, const size_t size);

}  // namespace ucxx
