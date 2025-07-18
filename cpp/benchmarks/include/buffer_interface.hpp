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

enum class MemoryType {
  Host,
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  Cuda,
  CudaManaged,
  CudaAsync,
#endif
};

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

// Forward declaration of allocation function
BufferMapPtr allocateTransferBuffers(size_t message_size);

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

  // Static factory method to create host buffer interface
  static std::unique_ptr<HostBufferInterface> createBufferInterface(size_t message_size, bool reuse_alloc)
  {
    static BufferMapPtr reuseBuffer;

    if (reuse_alloc) {
      if (!reuseBuffer) {
        reuseBuffer = allocateTransferBuffers(message_size);
      }
      return std::make_unique<HostBufferInterface>(reuseBuffer);
    } else {
      auto bufferMap = allocateTransferBuffers(message_size);
      return std::make_unique<HostBufferInterface>(bufferMap);
    }
  }
};

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
// Unified CUDA buffer interface that all CUDA buffer types implement
struct CudaBufferInterfaceBase : public BufferInterface {
  virtual ~CudaBufferInterfaceBase() = default;

  // Pure virtual methods that derived classes must implement
  virtual void* getBufferPtr(transfer_type_t type) = 0;
  virtual std::shared_ptr<void> getBufferMap() = 0;
  virtual void initializeBuffer(transfer_type_t type, size_t message_size) = 0;
  virtual void verifyBufferResults(size_t message_size) = 0;

  // Common implementation for getSendPtr and getRecvPtr
  void* getSendPtr() override { return getBufferPtr(SEND); }
  void* getRecvPtr() override { return getBufferPtr(RECV); }

  // Common verification logic
  void verifyResults(size_t message_size) override
  {
    verifyBufferResults(message_size);
  }

  // Static factory method to create appropriate buffer interface
  static std::unique_ptr<CudaBufferInterfaceBase> createBufferInterface(MemoryType memory_type, size_t message_size, bool reuse_alloc);

  // Static allocation method that each derived class must implement
  static std::unique_ptr<CudaBufferInterfaceBase> allocateBuffers(MemoryType memory_type, size_t message_size);

  // Virtual clone method for reuse buffers
  virtual std::unique_ptr<CudaBufferInterfaceBase> clone() const = 0;
};

// Template-based CUDA buffer interface implementation
template<typename BufferType>
struct CudaBufferInterface : public CudaBufferInterfaceBase {
  std::shared_ptr<std::unordered_map<transfer_type_t, BufferType>> bufferMap;

  explicit CudaBufferInterface(std::shared_ptr<std::unordered_map<transfer_type_t, BufferType>> buf)
    : bufferMap(buf)
  {
  }

  void* getBufferPtr(transfer_type_t type) override { return (*bufferMap)[type].ptr; }

  std::shared_ptr<void> getBufferMap() override { return bufferMap; }

  void initializeBuffer(transfer_type_t type, size_t message_size) override
  {
    if (type == SEND) {
      std::vector<char> pattern(message_size, 0xaa);
      if constexpr (std::is_same_v<BufferType, CudaAsyncBuffer>) {
        CUDA_EXIT_ON_ERROR(cudaMemcpyAsync((*bufferMap)[SEND].ptr,
                                           pattern.data(),
                                           message_size,
                                           cudaMemcpyHostToDevice,
                                           (*bufferMap)[SEND].stream),
                           "CUDA async send buffer initialization");
        CUDA_EXIT_ON_ERROR(cudaStreamSynchronize((*bufferMap)[SEND].stream),
                           "CUDA stream synchronization");
      } else if constexpr (std::is_same_v<BufferType, CudaManagedBuffer>) {
        std::memcpy((*bufferMap)[SEND].ptr, pattern.data(), message_size);
      } else {
        CUDA_EXIT_ON_ERROR(
          cudaMemcpy((*bufferMap)[SEND].ptr, pattern.data(), message_size, cudaMemcpyHostToDevice),
          "CUDA send buffer initialization");
      }
    }
  }

  void verifyBufferResults(size_t message_size) override
  {
    std::vector<char> send_data(message_size);
    std::vector<char> recv_data(message_size);

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
    } else if constexpr (std::is_same_v<BufferType, CudaManagedBuffer>) {
      // Managed memory can be accessed directly from host
      std::memcpy(send_data.data(), (*bufferMap)[SEND].ptr, message_size);
      std::memcpy(recv_data.data(), (*bufferMap)[RECV].ptr, message_size);
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

  // Static allocation methods
  static std::shared_ptr<std::unordered_map<transfer_type_t, BufferType>> allocateBuffers(size_t message_size);

  // Clone method implementation
  std::unique_ptr<CudaBufferInterfaceBase> clone() const override
  {
    return std::make_unique<CudaBufferInterface<BufferType>>(bufferMap);
  }
};
#endif

// Buffer allocation functions (always available)
BufferMapPtr allocateTransferBuffers(size_t message_size)
{
  return std::make_shared<BufferMap>(BufferMap{{SEND, std::vector<char>(message_size, 0xaa)},
                                               {RECV, std::vector<char>(message_size)}});
}

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
// Type aliases for convenience
using CudaDeviceBufferInterface = CudaBufferInterface<CudaBuffer>;
using CudaManagedBufferInterface = CudaBufferInterface<CudaManagedBuffer>;
using CudaAsyncBufferInterface = CudaBufferInterface<CudaAsyncBuffer>;

// Template specialization implementations for allocation methods
template<>
std::shared_ptr<CudaBufferMap> CudaBufferInterface<CudaBuffer>::allocateBuffers(size_t message_size)
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

template<>
std::shared_ptr<CudaManagedBufferMap> CudaBufferInterface<CudaManagedBuffer>::allocateBuffers(size_t message_size)
{
  auto bufferMap     = std::make_shared<CudaManagedBufferMap>();
  (*bufferMap)[SEND] = CudaManagedBuffer(message_size);
  (*bufferMap)[RECV] = CudaManagedBuffer(message_size);

  // Initialize send buffer with pattern (managed memory can be accessed from host)
  std::vector<char> pattern(message_size, 0xaa);
  std::memcpy((*bufferMap)[SEND].ptr, pattern.data(), message_size);

  return bufferMap;
}

template<>
std::shared_ptr<CudaAsyncBufferMap> CudaBufferInterface<CudaAsyncBuffer>::allocateBuffers(size_t message_size)
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

// Factory method implementation
std::unique_ptr<CudaBufferInterfaceBase> CudaBufferInterfaceBase::createBufferInterface(MemoryType memory_type, size_t message_size, bool reuse_alloc)
{
  if (reuse_alloc) {
    // Use static reuse buffers for each memory type
    static std::unordered_map<MemoryType, std::unique_ptr<CudaBufferInterfaceBase>> reuseBuffers;

    // Check if we already have a reuse buffer for this memory type
    auto it = reuseBuffers.find(memory_type);
    if (it == reuseBuffers.end()) {
      // Create new reuse buffer
      reuseBuffers[memory_type] = allocateBuffers(memory_type, message_size);
    }

    // Clone the reuse buffer (we need to implement a clone method)
    return reuseBuffers[memory_type]->clone();
  } else {
    // Create new buffer
    return allocateBuffers(memory_type, message_size);
  }
}

// Unified allocation method implementation
std::unique_ptr<CudaBufferInterfaceBase> CudaBufferInterfaceBase::allocateBuffers(MemoryType memory_type, size_t message_size)
{
  switch (memory_type) {
    case MemoryType::Cuda: {
      auto bufferMap = CudaDeviceBufferInterface::allocateBuffers(message_size);
      return std::make_unique<CudaDeviceBufferInterface>(bufferMap);
    }
    case MemoryType::CudaManaged: {
      auto bufferMap = CudaManagedBufferInterface::allocateBuffers(message_size);
      return std::make_unique<CudaManagedBufferInterface>(bufferMap);
    }
    case MemoryType::CudaAsync: {
      auto bufferMap = CudaAsyncBufferInterface::allocateBuffers(message_size);
      return std::make_unique<CudaAsyncBufferInterface>(bufferMap);
    }
    default:
      throw std::runtime_error("Unsupported CUDA memory type");
  }
}
#endif
