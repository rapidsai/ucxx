/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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

  // TEMPORARY DEBUGGING SWITCH (do not merge): UCXX_DEBUG_CCCL_BUFFER_ALLOCATOR=cudamalloc makes
  // the buffer a plain cudaMalloc/cudaFree allocation instead of an allocation from CCCL's
  // stream-ordered default device memory pool. Used to find out whether the allocator matters for
  // the hang in cuMemRetainAllocationHandle seen on V100 + driver 580.
  static bool use_cuda_malloc()
  {
    static const bool value = [] {
      const char* env = std::getenv("UCXX_DEBUG_CCCL_BUFFER_ALLOCATOR");
      const bool v    = env != nullptr && std::strcmp(env, "cudamalloc") == 0;
      std::fprintf(stderr,
                   "ucxx: DEBUG CCCLBuffer allocator = %s (UCXX_DEBUG_CCCL_BUFFER_ALLOCATOR=%s)\n",
                   v ? "cudaMalloc" : "cuda::device_default_memory_pool",
                   env != nullptr ? env : "<unset>");
      return v;
    }();
    return value;
  }

  std::optional<cccl_buffer_type> buffer{};  // pool-backed buffer (default)
  void* raw{nullptr};                        // cudaMalloc-backed buffer (debug switch)

  // CCCL's cuda::device_default_memory_pool() requires an active CUDA primary context.
  // cudaFree(0) is the standard zero-cost idiom to initialize it.
  static auto get_device_pool()
  {
    cudaFree(0);  // Ensure CUDA primary context is initialized
    return ::cuda::device_default_memory_pool(::cuda::device_ref{0});
  }

  explicit CCCLBufferImpl(const size_t size)
  {
    if (use_cuda_malloc()) {
      cudaFree(0);  // Ensure CUDA primary context is initialized
      const cudaError_t err = cudaMalloc(&raw, size);
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
      }
    } else {
      buffer.emplace(::cudaStream_t{0}, get_device_pool(), size, ::cuda::no_init);
    }
  }

  ~CCCLBufferImpl()
  {
    if (raw != nullptr) cudaFree(raw);
  }

  CCCLBufferImpl(const CCCLBufferImpl&)            = delete;
  CCCLBufferImpl& operator=(const CCCLBufferImpl&) = delete;

  void* data()
  {
    // Explicit cast required: cuda::buffer::data() returns cuda::std::byte*, not void*.
    return raw != nullptr ? raw : static_cast<void*>(buffer->data());
  }
};

CCCLBuffer::CCCLBuffer(const size_t size)
  : Buffer(BufferType::CCCL, size), _impl{std::make_unique<CCCLBufferImpl>(size)}
{
  ucxx_trace_data("ucxx::CCCLBuffer created: %p, impl: %p, size: %lu", this, _impl.get(), size);
}

CCCLBuffer::~CCCLBuffer() = default;

void* CCCLBuffer::data()
{
  ucxx_trace_data("ucxx::CCCLBuffer::%s, CCCLBuffer: %p, impl: %p", __func__, this, _impl.get());
  if (!_impl) throw std::runtime_error("Invalid object or already released");

  return _impl->data();
}

}  // namespace ucxx
