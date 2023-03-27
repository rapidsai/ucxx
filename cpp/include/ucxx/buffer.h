/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
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
  BufferType _bufferType{BufferType::Invalid};  ///< Buffer type
  size_t _size;                                 ///< Buffer size

  /**
   * @brief Protected constructor of abstract type `Buffer`.
   *
   * This is the constructor that should be called by derived classes to store
   * general information about the buffer, such as its type and size.
   *
   * @param[in] bufferType the type of buffer the object holds.
   * @param[in] size the size of the contained buffer.
   */
  Buffer(const BufferType bufferType, const size_t size);

 public:
  Buffer()              = delete;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(Buffer const&) = delete;
  Buffer(Buffer&& o)               = delete;
  Buffer& operator=(Buffer&& o) = delete;

  /**
   * @brief Virtual destructor.
   *
   * Virtual destructor with empty implementation.
   */
  virtual ~Buffer();

  /**
   * @brief Get the type of buffer the object holds.
   *
   * The type of buffer the object holds is important to ensure proper casting
   * of the object into the correct derived type.
   *
   * @return the type of buffer the object holds
   */
  BufferType getType() const noexcept;

  /**
   * @brief Get the size of the contained buffer.
   *
   * The size in bytes of the contained buffer.
   *
   * @return the size of the contained buffer.
   */
  size_t getSize() const noexcept;

  /**
   * @brief Abstract method returning void pointer to buffer.
   *
   * Get the void pointer to the underlying buffer that holds the data. This
   * is meant to return the actual allocation, and not a pointer to some
   * container to the buffer it holds.
   *
   * @return the void pointer to the buffer.
   */
  virtual void* data() = 0;
};

class HostBuffer : public Buffer {
 private:
  void* _buffer;  ///< Pointer to the allocated buffer

 public:
  HostBuffer()                  = delete;
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(HostBuffer const&) = delete;
  HostBuffer(HostBuffer&& o)               = delete;
  HostBuffer& operator=(HostBuffer&& o) = delete;

  /**
   * @brief Constructor of concrete type `HostBuffer`.
   *
   * Constructor to materialize a buffer holding host memory. The internal buffer
   * is allocated using `malloc`, and thus should be freed with `free`.
   *
   * @param[in] size the size of the host buffer to allocate.
   *
   * @code{.cpp}
   * // Allocate host buffer of 1KiB
   * auto buffer = HostBuffer(1024);
   * @endcode
   */
  HostBuffer(const size_t size);

  /**
   * @brief Destructor of concrete type `HostBuffer`.
   *
   * Frees the underlying buffer, unless the underlying buffer was released to
   * the user after a call to `release`.
   */
  ~HostBuffer();

  /**
   * @brief Release the allocated host buffer to the caller.
   *
   * Release ownership of the buffer to the caller. After this method is called,
   * the caller becomes responsible for its deallocation once it is not needed
   * anymore. The buffer is allocated with `malloc`, and should be properly
   * disposed of by a call to `free`.
   *
   * The original `HostBuffer` object becomes invalid.
   *
   * @code{.cpp}
   * // Allocate host buffer of 1KiB
   * auto buffer = HostBuffer(1024);
   * void* bufferPtr = buffer.release();
   *
   * // do work on bufferPtr
   *
   * // Free buffer
   * free(bufferPtr);
   * @endcode
   *
   * @throws std::runtime_error if object has been released.
   *
   * @return the void pointer to the buffer.
   */
  void* release();

  /**
   * @brief Get a pointer to the allocated raw host buffer.
   *
   * Get a pointer to the underlying buffer, but does not release ownership.
   *
   * @code{.cpp}
   * // Allocate host buffer of 1KiB
   * auto buffer = HostBuffer(1024);
   * void* bufferPtr = buffer.data();
   *
   * // do work on bufferPtr
   *
   * // Memory is freed once `buffer` goes out-of-scope.
   * @endcode
   *
   * @throws std::runtime_error if object has been released.
   *
   * @return the void pointer to the buffer.
   */
  virtual void* data();
};

#if UCXX_ENABLE_RMM
class RMMBuffer : public Buffer {
 private:
  std::unique_ptr<rmm::device_buffer> _buffer;  ///< RMM-allocated device buffer

 public:
  RMMBuffer()                 = delete;
  RMMBuffer(const RMMBuffer&) = delete;
  RMMBuffer& operator=(RMMBuffer const&) = delete;
  RMMBuffer(RMMBuffer&& o)               = delete;
  RMMBuffer& operator=(RMMBuffer&& o) = delete;

  /**
   * @brief Constructor of concrete type `RMMBuffer`.
   *
   * Constructor to materialize a buffer holding device memory. The internal
   * buffer holds a `std::unique_ptr<rmm::device_buffer>` and is destroyed
   * when the object goes out-of-scope or is explicitly deleted.
   *
   * @param[in] size the size of the device buffer to allocate.
   *
   * @code{.cpp}
   * // Allocate host buffer of 1KiB
   * auto buffer = RMMBuffer(1024);
   * @endcode
   */
  RMMBuffer(const size_t size);

  /**
   * @brief Release the allocated `rmm::device_buffer` to the caller.
   *
   * Release ownership of the `rmm::device_buffer` to the caller. After this
   * method is called, the caller becomes responsible for the destruction of
   * the object once it is not needed anymore. The `rmm::device_buffer` is held
   * owned by the `unique_ptr` and will be deallocated once it goes out-of-scope
   * or gets explicitly deleted.
   *
   * The original `RMMBuffer` object becomes invalid.
   *
   * @code{.cpp}
   * // Allocate RMM buffer of 1KiB
   * auto buffer = RMMBuffer(1024);
   * std::unique_ptr<RMMBuffer> rmmBuffer= buffer.release();
   *
   * // do work on rmmBuffer
   *
   * // `rmm::device_buffer` is destroyed and device Memory is freed once
   * // `rmmBuffer` goes out-of-scope.
   * @endcode
   *
   * @throws std::runtime_error if object has been released.
   *
   * @return the void pointer to the buffer.
   */
  std::unique_ptr<rmm::device_buffer> release();

  /**
   * @brief Get a pointer to the allocated raw device buffer.
   *
   * Get a pointer to the underlying buffer, but does not release ownership.
   *
   * @code{.cpp}
   * // Allocate device buffer of 1KiB
   * auto buffer = RMMBuffer(1024);
   * void* bufferPtr = buffer.data();
   *
   * // do work on bufferPtr
   *
   * // `rmm::device_buffer` is destroyed and device Memory is freed once
   * // `buffer` goes out-of-scope.
   * @endcode
   *
   * @throws std::runtime_error if object has been released.
   *
   * @return the void pointer to the device buffer.
   */
  virtual void* data();
};
#endif

Buffer* allocateBuffer(BufferType bufferType, const size_t size);

}  // namespace ucxx
