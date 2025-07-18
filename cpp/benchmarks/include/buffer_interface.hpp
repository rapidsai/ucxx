/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <ucxx/api.h>

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
#include <cuda_runtime.h>

// CUDA error checking macro (if not already defined)
#ifndef CUDA_EXIT_ON_ERROR
#define CUDA_EXIT_ON_ERROR(operation, context)                                                  \
  ([&]() {                                                                                      \
    cudaError_t err = operation;                                                                \
    if (err != cudaSuccess) {                                                                   \
      std::cerr << "CUDA error in " << context << " at " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(err) << std::endl;                                        \
      std::exit(-1);                                                                            \
    }                                                                                           \
    return err;                                                                                 \
  })()
#endif
#endif

enum transfer_type_t { SEND, RECV };

typedef std::unordered_map<transfer_type_t, std::vector<char>> BufferMap;
typedef std::shared_ptr<BufferMap> BufferMapPtr;

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
// CUDA memory buffer structure (conditional)
struct CudaBuffer {
  void* ptr{nullptr};
  size_t size{0};

  CudaBuffer() = default;
  explicit CudaBuffer(size_t buffer_size) : size(buffer_size)
  {
    CUDA_EXIT_ON_ERROR(cudaMalloc(&ptr, size), "CUDA memory allocation");
  }

  ~CudaBuffer()
  {
    if (ptr) { cudaFree(ptr); }
  }

  CudaBuffer(const CudaBuffer&)            = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
  CudaBuffer(CudaBuffer&& other) noexcept : ptr(other.ptr), size(other.size)
  {
    other.ptr  = nullptr;
    other.size = 0;
  }
  CudaBuffer& operator=(CudaBuffer&& other) noexcept
  {
    if (this != &other) {
      if (ptr) cudaFree(ptr);
      ptr        = other.ptr;
      size       = other.size;
      other.ptr  = nullptr;
      other.size = 0;
    }
    return *this;
  }
};

// CUDA managed memory buffer structure
struct CudaManagedBuffer {
  void* ptr{nullptr};
  size_t size{0};

  CudaManagedBuffer() = default;
  explicit CudaManagedBuffer(size_t buffer_size) : size(buffer_size)
  {
    CUDA_EXIT_ON_ERROR(cudaMallocManaged(&ptr, size), "CUDA managed memory allocation");
  }

  ~CudaManagedBuffer()
  {
    if (ptr) { cudaFree(ptr); }
  }

  CudaManagedBuffer(const CudaManagedBuffer&)            = delete;
  CudaManagedBuffer& operator=(const CudaManagedBuffer&) = delete;
  CudaManagedBuffer(CudaManagedBuffer&& other) noexcept : ptr(other.ptr), size(other.size)
  {
    other.ptr  = nullptr;
    other.size = 0;
  }
  CudaManagedBuffer& operator=(CudaManagedBuffer&& other) noexcept
  {
    if (this != &other) {
      if (ptr) cudaFree(ptr);
      ptr        = other.ptr;
      size       = other.size;
      other.ptr  = nullptr;
      other.size = 0;
    }
    return *this;
  }
};

// CUDA async memory buffer structure
struct CudaAsyncBuffer {
  void* ptr{nullptr};
  size_t size{0};
  cudaStream_t stream{nullptr};

  CudaAsyncBuffer() = default;
  explicit CudaAsyncBuffer(size_t buffer_size) : size(buffer_size)
  {
    CUDA_EXIT_ON_ERROR(cudaStreamCreate(&stream), "CUDA stream creation");
    CUDA_EXIT_ON_ERROR(cudaMallocAsync(&ptr, size, stream), "CUDA async memory allocation");
  }

  ~CudaAsyncBuffer()
  {
    if (ptr) { cudaFreeAsync(ptr, stream); }
    if (stream) { cudaStreamDestroy(stream); }
  }

  CudaAsyncBuffer(const CudaAsyncBuffer&)            = delete;
  CudaAsyncBuffer& operator=(const CudaAsyncBuffer&) = delete;
  CudaAsyncBuffer(CudaAsyncBuffer&& other) noexcept
    : ptr(other.ptr), size(other.size), stream(other.stream)
  {
    other.ptr    = nullptr;
    other.size   = 0;
    other.stream = nullptr;
  }
  CudaAsyncBuffer& operator=(CudaAsyncBuffer&& other) noexcept
  {
    if (this != &other) {
      if (ptr) cudaFreeAsync(ptr, stream);
      if (stream) cudaStreamDestroy(stream);
      ptr          = other.ptr;
      size         = other.size;
      stream       = other.stream;
      other.ptr    = nullptr;
      other.size   = 0;
      other.stream = nullptr;
    }
    return *this;
  }
};

typedef std::unordered_map<transfer_type_t, CudaBuffer> CudaBufferMap;
typedef std::unordered_map<transfer_type_t, CudaManagedBuffer> CudaManagedBufferMap;
typedef std::unordered_map<transfer_type_t, CudaAsyncBuffer> CudaAsyncBufferMap;
typedef std::shared_ptr<CudaBufferMap> CudaBufferMapPtr;
typedef std::shared_ptr<CudaManagedBufferMap> CudaManagedBufferMapPtr;
typedef std::shared_ptr<CudaAsyncBufferMap> CudaAsyncBufferMapPtr;
#endif

typedef std::unordered_map<transfer_type_t, ucxx::Tag> TagMap;
typedef std::shared_ptr<TagMap> TagMapPtr;

// Unified buffer interface for different memory types
struct BufferInterface {
  virtual void* getSendPtr()                      = 0;
  virtual void* getRecvPtr()                      = 0;
  virtual void verifyResults(size_t message_size) = 0;
  virtual ~BufferInterface()                      = default;
};

// Host memory buffer implementation
struct HostBufferInterface : public BufferInterface {
  BufferMapPtr bufferMap;

  explicit HostBufferInterface(BufferMapPtr buf) : bufferMap(buf) {}

  void* getSendPtr() override { return (*bufferMap)[SEND].data(); }
  void* getRecvPtr() override { return (*bufferMap)[RECV].data(); }

  void verifyResults(size_t message_size) override
  {
    for (size_t j = 0; j < (*bufferMap)[SEND].size(); ++j)
      assert((*bufferMap)[RECV][j] == (*bufferMap)[SEND][j]);
  }
};

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
// Template-based CUDA buffer interface to reduce duplication
template<typename BufferType>
struct CudaBufferInterface : public BufferInterface {
  std::shared_ptr<std::unordered_map<transfer_type_t, BufferType>> bufferMap;

  explicit CudaBufferInterface(std::shared_ptr<std::unordered_map<transfer_type_t, BufferType>> buf)
    : bufferMap(buf)
  {
  }

  void* getSendPtr() override { return (*bufferMap)[SEND].ptr; }
  void* getRecvPtr() override { return (*bufferMap)[RECV].ptr; }

  void verifyResults(size_t message_size) override
  {
    std::vector<char> send_data(message_size);
    std::vector<char> recv_data(message_size);

    // Use async copy for all types, with appropriate stream handling
    if constexpr (std::is_same_v<BufferType, CudaAsyncBuffer>) {
      // For async buffers, use the stream from the buffer
      CUDA_EXIT_ON_ERROR(
        cudaMemcpyAsync(send_data.data(), (*bufferMap)[SEND].ptr, message_size, cudaMemcpyDeviceToHost,
                       (*bufferMap)[SEND].stream),
        "CUDA async send data copy for verification");
      CUDA_EXIT_ON_ERROR(
        cudaMemcpyAsync(recv_data.data(), (*bufferMap)[RECV].ptr, message_size, cudaMemcpyDeviceToHost,
                       (*bufferMap)[RECV].stream),
        "CUDA async recv data copy for verification");

      // Synchronize streams
      CUDA_EXIT_ON_ERROR(cudaStreamSynchronize((*bufferMap)[SEND].stream),
                         "CUDA send stream synchronization for verification");
      CUDA_EXIT_ON_ERROR(cudaStreamSynchronize((*bufferMap)[RECV].stream),
                         "CUDA recv stream synchronization for verification");
    } else {
      // For non-async buffers, use synchronous copy with null stream
      CUDA_EXIT_ON_ERROR(
        cudaMemcpyAsync(send_data.data(), (*bufferMap)[SEND].ptr, message_size, cudaMemcpyDeviceToHost,
                       nullptr),
        "CUDA send data copy for verification");
      CUDA_EXIT_ON_ERROR(
        cudaMemcpyAsync(recv_data.data(), (*bufferMap)[RECV].ptr, message_size, cudaMemcpyDeviceToHost,
                       nullptr),
        "CUDA recv data copy for verification");
    }

    for (size_t j = 0; j < send_data.size(); ++j)
      assert(recv_data[j] == send_data[j]);
  }
};

// Specialization for managed memory (no copy needed)
template<>
struct CudaBufferInterface<CudaManagedBuffer> : public BufferInterface {
  CudaManagedBufferMapPtr bufferMap;

  explicit CudaBufferInterface(CudaManagedBufferMapPtr buf) : bufferMap(buf) {}

  void* getSendPtr() override { return (*bufferMap)[SEND].ptr; }
  void* getRecvPtr() override { return (*bufferMap)[RECV].ptr; }

  void verifyResults(size_t message_size) override
  {
    std::vector<char> send_data(message_size);
    std::vector<char> recv_data(message_size);

    // Managed memory can be accessed directly from host
    std::memcpy(send_data.data(), (*bufferMap)[SEND].ptr, message_size);
    std::memcpy(recv_data.data(), (*bufferMap)[RECV].ptr, message_size);

    for (size_t j = 0; j < send_data.size(); ++j)
      assert(recv_data[j] == send_data[j]);
  }
};

// Type aliases for convenience
using CudaDeviceBufferInterface = CudaBufferInterface<CudaBuffer>;
using CudaManagedBufferInterface = CudaBufferInterface<CudaManagedBuffer>;
using CudaAsyncBufferInterface = CudaBufferInterface<CudaAsyncBuffer>;

// Buffer allocation functions
BufferMapPtr allocateTransferBuffers(size_t message_size)
{
  return std::make_shared<BufferMap>(BufferMap{{SEND, std::vector<char>(message_size, 0xaa)},
                                               {RECV, std::vector<char>(message_size)}});
}

CudaBufferMapPtr allocateCudaTransferBuffers(size_t message_size)
{
  auto bufferMap     = std::make_shared<CudaBufferMap>();
  (*bufferMap)[SEND] = CudaBuffer(message_size);
  (*bufferMap)[RECV] = CudaBuffer(message_size);

  // Initialize send buffer with pattern
  std::vector<char> pattern(message_size, 0xaa);
  CUDA_EXIT_ON_ERROR(
    cudaMemcpy((*bufferMap)[SEND].ptr, pattern.data(), message_size, cudaMemcpyHostToDevice),
    "CUDA send buffer initialization");

  return bufferMap;
}

CudaManagedBufferMapPtr allocateCudaManagedTransferBuffers(size_t message_size)
{
  auto bufferMap     = std::make_shared<CudaManagedBufferMap>();
  (*bufferMap)[SEND] = CudaManagedBuffer(message_size);
  (*bufferMap)[RECV] = CudaManagedBuffer(message_size);

  // Initialize send buffer with pattern (managed memory can be accessed from host)
  std::vector<char> pattern(message_size, 0xaa);
  std::memcpy((*bufferMap)[SEND].ptr, pattern.data(), message_size);

  return bufferMap;
}

CudaAsyncBufferMapPtr allocateCudaAsyncTransferBuffers(size_t message_size)
{
  auto bufferMap     = std::make_shared<CudaAsyncBufferMap>();
  (*bufferMap)[SEND] = CudaAsyncBuffer(message_size);
  (*bufferMap)[RECV] = CudaAsyncBuffer(message_size);

  // Initialize send buffer with pattern using async copy
  std::vector<char> pattern(message_size, 0xaa);
  CUDA_EXIT_ON_ERROR(cudaMemcpyAsync((*bufferMap)[SEND].ptr,
                                     pattern.data(),
                                     message_size,
                                     cudaMemcpyHostToDevice,
                                     (*bufferMap)[SEND].stream),
                     "CUDA async send buffer initialization");

  // Synchronize the stream to ensure the copy is complete
  CUDA_EXIT_ON_ERROR(cudaStreamSynchronize((*bufferMap)[SEND].stream),
                     "CUDA stream synchronization");

  return bufferMap;
}
#endif
