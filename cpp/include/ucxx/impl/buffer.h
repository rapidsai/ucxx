/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <iterator>

#include <ucxx/buffer.h>

#if UCXX_ENABLE_RMM
#include <rmm/device_buffer.hpp>
#endif

namespace ucxx {

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

void PyHostBuffer::free(void* ptr) { ::free(ptr); }

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
