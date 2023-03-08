/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <iterator>
#include <memory>
#include <utility>

#include <ucxx/buffer.h>

#if UCXX_ENABLE_RMM
#include <rmm/device_buffer.hpp>
#endif

namespace ucxx {

Buffer::Buffer(const BufferType bufferType, const size_t size)
  : _bufferType{bufferType}, _size{size}
{
}

Buffer::~Buffer() {}

BufferType Buffer::getType() const noexcept { return _bufferType; }

size_t Buffer::getSize() const noexcept { return _size; }

HostBuffer::HostBuffer(const size_t size) : Buffer(BufferType::Host, size), _buffer{malloc(size)}
{
  ucxx_trace_data("HostBuffer(%lu), _buffer: %p", size, _buffer);
}

HostBuffer::~HostBuffer()
{
  if (_buffer) free(_buffer);
}

void* HostBuffer::release()
{
  ucxx_trace_data("HostBuffer::release(), _buffer: %p", _buffer);
  if (!_buffer) throw std::runtime_error("Invalid object or already released");

  _bufferType = ucxx::BufferType::Invalid;
  _size       = 0;

  return std::exchange(_buffer, nullptr);
}

void* HostBuffer::data()
{
  ucxx_trace_data("HostBuffer::data(), _buffer: %p", _buffer);
  if (!_buffer) throw std::runtime_error("Invalid object or already released");

  return _buffer;
}

#if UCXX_ENABLE_RMM
RMMBuffer::RMMBuffer(const size_t size)
  : Buffer(BufferType::RMM, size),
    _buffer{std::make_unique<rmm::device_buffer>(size, rmm::cuda_stream_default)}
{
  ucxx_trace_data("RMMBuffer(%lu), _buffer: %p", size, _buffer.get());
}

std::unique_ptr<rmm::device_buffer> RMMBuffer::release()
{
  ucxx_trace_data("RMMBuffer::release(), _buffer: %p", _buffer.get());
  if (!_buffer) throw std::runtime_error("Invalid object or already released");

  _bufferType = ucxx::BufferType::Invalid;
  _size       = 0;

  return std::move(_buffer);
}

void* RMMBuffer::data()
{
  ucxx_trace_data("RMMBuffer::data(), _buffer: %p", _buffer.get());
  if (!_buffer) throw std::runtime_error("Invalid object or already released");

  return _buffer->data();
}
#endif

Buffer* allocateBuffer(const BufferType bufferType, const size_t size)
{
#if UCXX_ENABLE_RMM
  if (bufferType == BufferType::RMM)
    return new RMMBuffer(size);
  else
#else
  if (bufferType == BufferType::RMM)
    throw std::runtime_error("RMM support not enabled, please compile with -DUCXX_ENABLE_RMM=1");
#endif
    return new HostBuffer(size);
}

}  // namespace ucxx
