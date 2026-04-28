/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <stdexcept>
#include <utility>

#include <ucxx/buffer.h>
#include <ucxx/log.h>

#include <cuda/buffer>
#include <cuda/memory_resource>

#include <cuda_runtime_api.h>

namespace ucxx {

/**
 * @brief Concrete CCCL buffer implementation.
 */
struct CCCLBufferImpl {
  using cccl_buffer_type = ::cuda::buffer<::cuda::std::byte, ::cuda::mr::device_accessible>;
  cccl_buffer_type buffer;

  static auto get_device_pool() {
    cudaFree(0);  // Ensure CUDA primary context is initialized
    return ::cuda::device_default_memory_pool(::cuda::device_ref{0});
  }

  explicit CCCLBufferImpl(const size_t size)
    : buffer{::cudaStream_t{0}, get_device_pool(), size, ::cuda::no_init}
  {}
};

CCCLBuffer::CCCLBuffer(const size_t size)
  : Buffer(BufferType::CCCL, size), _impl{new CCCLBufferImpl(size)}
{
  ucxx_trace_data("ucxx::CCCLBuffer created: %p, impl: %p, size: %lu", this, _impl, size);
}

CCCLBuffer::CCCLBuffer(CCCLBufferImpl* impl)
  : Buffer(BufferType::CCCL, impl->buffer.size()), _impl{impl}
{
  ucxx_trace_data("ucxx::CCCLBuffer created: %p, impl: %p, size: %lu", this, _impl, _size);
}

CCCLBuffer::~CCCLBuffer()
{
  delete _impl;
  _impl = nullptr;
}

CCCLBufferImpl* CCCLBuffer::release()
{
  ucxx_trace_data("ucxx::CCCLBuffer::%s, CCCLBuffer: %p, _impl: %p", __func__, this, _impl);
  if (!_impl) throw std::runtime_error("Invalid object or already released");

  _bufferType = ucxx::BufferType::Invalid;
  _size       = 0;

  return std::exchange(_impl, nullptr);
}

void* CCCLBuffer::data()
{
  ucxx_trace_data("ucxx::CCCLBuffer::%s, CCCLBuffer: %p, impl: %p", __func__, this, _impl);
  if (!_impl) throw std::runtime_error("Invalid object or already released");

  return static_cast<void*>(_impl->buffer.data());
}

}  // namespace ucxx
