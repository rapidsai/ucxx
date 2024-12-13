/**
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint.h>
#include <ucxx/memory_handle.h>

namespace ucxx {

typedef size_t SerializedRemoteKeyHash;

/**
 * @brief Component holding a UCP rkey (remote key).
 *
 * To provide RMA (Remote Memory Access) to memory handles, UCP packs their information
 * in the form of `ucp_rkey_h` (remote key, or rkey for short). This class encapsulates
 * that object and provides methods to simplify its handling, both locally and remotely
 * including (de-)serialization for transfers over the wire and reconstruction of the
 * object on the remote process.
 */
class RemoteKey : public Component {
 private:
  ucp_rkey_h _remoteKey{nullptr};  ///< The unpacked remote key.
  void* _packedRemoteKey{
    nullptr};  ///< The packed ucp_rkey_h key, suitable for transfer to a remote process.
  size_t _packedRemoteKeySize{0};              ///< The size in bytes of the remote key.
  std::vector<char> _packedRemoteKeyVector{};  ///< The deserialized packed remote key.
  uint64_t _memoryBaseAddress{0};              ///< The allocation's base address.
  size_t _memorySize{0};                       ///< The actual allocation size.

  /**
   * @brief Private constructor of `ucxx::RemoteKey`.
   *
   * This is the internal implementation of `ucxx::RemoteKey` constructor from a local
   * `std::shared_ptr<ucxx::MemoryHandle>`, made private not to be called directly. This
   * constructor is made private to ensure all UCXX objects are shared pointers and the
   * correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::MemoryHandle::createRemoteKey()`
   * - `ucxx::createRemoteKeyFromMemoryHandle()`
   *
   * @param[in] memoryHandle the memory handle mapped on the local process.
   */
  explicit RemoteKey(std::shared_ptr<MemoryHandle> memoryHandle);

  /**
   * @brief Private constructor of `ucxx::RemoteKey`.
   *
   * This is the internal implementation of `ucxx::RemoteKey` constructor from a remote
   * `std::shared_ptr<ucxx::MemoryHandle>`, made private not to be called directly. This
   * constructor is made private to ensure all UCXX objects are shared pointers and the
   * correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::createRemoteKeyFromSerialized()`
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] serializedRemoteKey the remote key that was serialized by the owner of
   *                                the memory handle and transferred over-the-wire for
   *                                reconstruction and remote access.
   */
  RemoteKey(std::shared_ptr<Endpoint> endpoint, SerializedRemoteKey serializedRemoteKey);

  /**
   * @brief Deserialize and reconstruct the remote key.
   *
   * Deserialize the remote key that was serialized with `ucxx::RemoteKey::serialize()` and
   * possibly transferred over-the-wire and reconstruct the object to allow remote access.
   *
   * @code{.cpp}
   * // remoteKey is `std::shared_ptr<ucxx::RemoteKey>`
   * auto serializedRemoteKey = remoteKey->serialize();
   * @endcode
   *
   * @throws std::runtime_error if checksum of the serialized object fails.
   *
   * @returns The deserialized remote key.
   */
  void deserialize(const SerializedRemoteKey& serializedHeader);

 public:
  /**
   * @brief Constructor for `std::shared_ptr<ucxx::RemoteKey>` from local memory handle.
   *
   * The constructor for a `std::shared_ptr<ucxx::RemoteKey>` object from a local
   * `std::shared_ptr<ucxx::MemoryHandle>`, mapping a local memory buffer to be made
   * accessible from a remote endpoint to perform RMA (Remote Memory Access) on the memory.
   *
   * @code{.cpp}
   * // `memoryHandle` is `std::shared_ptr<ucxx::MemoryHandle>`
   * auto remoteKey = memoryHandle->createRemoteKey();
   *
   * // Equivalent to line above
   * // auto remoteKey = ucxx::createRemoteKeyFromMemoryHandle(memoryHandle);
   * @endcode
   *
   * @throws ucxx::Error if `ucp_rkey_pack` fails.
   *
   * @param[in] memoryHandle the memory handle mapped on the local process.
   *
   * @returns The `shared_ptr<ucxx::RemoteKey>` object
   */
  friend std::shared_ptr<RemoteKey> createRemoteKeyFromMemoryHandle(
    std::shared_ptr<MemoryHandle> memoryHandle);

  /**
   * @brief Constructor for `std::shared_ptr<ucxx::RemoteKey>` from remote.
   *
   * The constructor for a `std::shared_ptr<ucxx::RemoteKey>` object from a serialized
   * `std::shared_ptr<ucxx::RemoteKey>`, mapping a remote memory buffer to be made
   * accessible via a local endpoint to perform RMA (Remote Memory Access) on the memory.
   *
   * @code{.cpp}
   * // `serializedRemoteKey` is `ucxx::SerializedRemoteKey>`, created on a remote worker
   * // after a call to `ucxx::RemoteKey::serialize()` and transferred over-the-wire.
   * auto remoteKey = ucxx::createRemoteKeyFromSerialized(serializedRemoteKey);
   *
   * // Equivalent to line above
   * // auto remoteKey = ucxx::createRemoteKeyFromMemoryHandle(memoryHandle);
   * @endcode
   *
   * @throws ucxx::Error if `ucp_ep_rkey_unpack` fails.
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] serializedRemoteKey the remote key that was serialized by the owner of
   *                                the memory handle and transferred over-the-wire for
   *                                reconstruction and remote access.
   *
   * @returns The `shared_ptr<ucxx::RemoteKey>` object
   */
  friend std::shared_ptr<RemoteKey> createRemoteKeyFromSerialized(
    std::shared_ptr<Endpoint> endpoint, SerializedRemoteKey serializedRemoteKey);

  ~RemoteKey();

  /**
   * @brief Get the underlying `ucp_rkey_h` handle.
   *
   * Lifetime of the `ucp_rkey_h` handle is managed by the `ucxx::RemoteKey` object and
   * its ownership is non-transferrable. Once the `ucxx::RemoteKey` is destroyed the handle
   * becomes invalid and so does the address to the remote memory handle it points to, it is
   * the user's responsibility to ensure the owner's lifetime while using the handle.
   *
   * @code{.cpp}
   * // remoteKey is `std::shared_ptr<ucxx::RemoteKey>`
   * auto remoteKeyHandle = remoteKey->getHandle();
   * @endcode
   *
   * @returns The underlying `ucp_mem_h` handle.
   */
  [[nodiscard]] ucp_rkey_h getHandle();

  /**
   * @brief Get the size of the memory allocation.
   *
   * Get the size of the memory allocation the remote key packs, which is at least the
   * number of bytes specified with the `size` argument passed to `createMemoryHandle()`.
   *
   * @code{.cpp}
   * // remoteKey is `std::shared_ptr<ucxx::RemoteKey>`
   * auto remoteMemorySize = remoteKey->getSize();
   * @endcode
   *
   * @returns The size of the memory allocation.
   */
  [[nodiscard]] size_t getSize() const;

  /**
   * @brief Get the base address of the memory allocation.
   *
   * Get the base address of the memory allocation the remote key packs, which is going
   * to be used as the remote address to put or get memory via the
   * `ucxx::Endpoint::memPut()` or `ucxx::Endpoint::memGet()` methods.
   *
   * @code{.cpp}
   * // remoteKey is `std::shared_ptr<ucxx::RemoteKey>`
   * auto remoteMemoryBaseAddress = remoteKey->getBaseAddress();
   * @endcode
   *
   * @returns The base address of the memory allocation.
   */
  [[nodiscard]] uint64_t getBaseAddress();

  /**
   * @brief Serialize the remote key.
   *
   * Serialize the remote key to allow over-the-wire transfer and subsequent
   * reconstruction of the object in the remote process.
   *
   * @code{.cpp}
   * // remoteKey is `std::shared_ptr<ucxx::RemoteKey>`
   * auto serializedRemoteKey = remoteKey->serialize();
   * @endcode
   *
   * @returns The serialized remote key.
   */
  [[nodiscard]] SerializedRemoteKey serialize() const;
};

}  // namespace ucxx
