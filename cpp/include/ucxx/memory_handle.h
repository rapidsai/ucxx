/**
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <string>
#include <utility>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/context.h>

namespace ucxx {

class RemoteKey;

/**
 * @brief Component holding a UCP memory handle.
 *
 * The UCP layer provides RMA (Remote Memory Access) to memory handles that it controls
 * in form of `ucp_mem_h` object, this class encapsulates that object and provides
 * methods to simplify its handling.
 */
class MemoryHandle : public Component {
 private:
  ucp_mem_h _handle{};       ///< The UCP handle to the memory allocation.
  size_t _size{0};           ///< The actual allocation size.
  uint64_t _baseAddress{0};  ///< The allocation's base address.
  ucs_memory_type_t _memoryType{
    UCS_MEMORY_TYPE_HOST};  ///< The memory type of the underlying allocation.

  /**
   * @brief Private constructor of `ucxx::MemoryHandle`.
   *
   * This is the internal implementation of `ucxx::MemoryHandle` constructor, made private
   * not to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Context::createMemoryHandle`
   * - `ucxx::createMemoryHandle()`
   *
   * @throws ucxx::Error if either `ucp_mem_map` or `ucp_mem_query` fail.
   *
   * @param[in] context     parent context where to map memory.
   * @param[in] size        the minimum size of the memory allocation
   * @param[in] buffer      the pointer to an existing allocation or `nullptr` to allocate a
   *                        new memory region.
   * @param[in] memoryType  the type of memory the handle points to.
   */
  MemoryHandle(std::shared_ptr<Context> context,
               const size_t size,
               void* buffer,
               const ucs_memory_type_t memoryType);

 public:
  MemoryHandle()                               = delete;
  MemoryHandle(const MemoryHandle&)            = delete;
  MemoryHandle& operator=(MemoryHandle const&) = delete;
  MemoryHandle(MemoryHandle&& o)               = delete;
  MemoryHandle& operator=(MemoryHandle&& o)    = delete;

  /**
   * @brief Constructor for `shared_ptr<ucxx::MemoryHandle>`.
   *
   * The constructor for a `shared_ptr<ucxx::MemoryHandle>` object, mapping a memory buffer
   * with UCP to provide RMA (Remote Memory Access) to.
   *
   * The allocation requires a `size` and a `buffer`.  The `buffer` provided may be either
   * a `nullptr`, in which case UCP will allocate a new memory region for it, or an already
   * existing allocation, in which case UCP will only map it for RMA and it's the caller's
   * responsibility to keep `buffer` alive until this object is destroyed. When the UCP
   * allocates `buffer` (i.e., when the value passed is `nullptr`), the actual size of the
   * allocation may be larger than requested, and can later be found calling the `getSize()`
   * method, if a preallocated buffer is passed `getSize()` will return the same value
   * specified for `size`.
   *
   * @code{.cpp}
   * // `context` is `std::shared_ptr<ucxx::Context>`
   * // Allocate a 128-byte buffer with UCP.
   * auto memoryHandle = context->createMemoryHandle(128, nullptr);
   *
   * // Equivalent to line above
   * // auto memoryHandle = ucxx::createMemoryHandle(context, 128, nullptr);
   *
   * // Map an existing 128-byte buffer with UCP.
   * size_t allocationSize = 128;
   * auto buffer = new uint8_t[allocationSize];
   * auto memoryHandleFromBuffer = context->createMemoryHandle(
   *    allocationSize * sizeof(*buffer), reinterpret_cast<void*>(buffer)
   * );
   *
   * // Equivalent to line above
   * // auto memoryHandleFromBuffer = ucxx::createMemoryHandle(
   * //    context, allocationSize * sizeof(*buffer), reinterpret_cast<void*>(buffer)
   * // );
   * @endcode
   *
   * @throws ucxx::Error if either `ucp_mem_map` or `ucp_mem_query` fail.
   *
   * @param[in] context     parent context where to map memory.
   * @param[in] size        the minimum size of the memory allocation
   * @param[in] buffer      the pointer to an existing allocation or `nullptr` to allocate a
   *                        new memory region.
   * @param[in] memoryType  the type of memory the handle points to.
   *
   * @returns The `shared_ptr<ucxx::MemoryHandle>` object
   */
  friend std::shared_ptr<MemoryHandle> createMemoryHandle(std::shared_ptr<Context> context,
                                                          const size_t size,
                                                          void* buffer,
                                                          const ucs_memory_type_t memoryType);

  ~MemoryHandle();

  /**
   * @brief Get the underlying `ucp_mem_h` handle.
   *
   * Lifetime of the `ucp_mem_h` handle is managed by the `ucxx::MemoryHandle` object and
   * its ownership is non-transferrable. Once the `ucxx::MemoryHandle` is destroyed the
   * memory is unmapped and the handle is not valid anymore, it is the user's responsibility
   * to ensure the owner's lifetime while using the handle.
   *
   * @code{.cpp}
   * // memoryHandle is `std::shared_ptr<ucxx::MemoryHandle>`
   * ucp_mem_h ucpMemoryHandle = memoryHandle->getHandle();
   * @endcode
   *
   * @returns The underlying `ucp_mem_h` handle.
   */
  [[nodiscard]] ucp_mem_h getHandle();

  /**
   * @brief Get the size of the memory allocation.
   *
   * Get the size of the memory allocation, which is at least the number of bytes specified
   * with the `size` argument passed to `createMemoryHandle()`.
   *
   * @code{.cpp}
   * // memoryHandle is `std::shared_ptr<ucxx::MemoryHandle>`
   * auto memorySize = memoryHandle->getSize();
   * @endcode
   *
   * @returns The size of the memory allocation.
   */
  [[nodiscard]] size_t getSize() const;

  /**
   * @brief Get the base address of the memory allocation.
   *
   * Get the base address of the memory allocation, which is going to be used as the remote
   * address to put or get memory via the `ucxx::Endpoint::memPut()` or
   * `ucxx::Endpoint::memGet()` methods.
   *
   * @code{.cpp}
   * // memoryHandle is `std::shared_ptr<ucxx::MemoryHandle>`
   * auto memoryBase Address = memoryHandle->getBaseAddress();
   * @endcode
   *
   * @returns The base address of the memory allocation.
   */
  [[nodiscard]] uint64_t getBaseAddress();

  [[nodiscard]] ucs_memory_type_t getMemoryType();

  [[nodiscard]] std::shared_ptr<RemoteKey> createRemoteKey();
};

}  // namespace ucxx
