/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/worker.h>

namespace ucxx {

/**
 * @brief Component encapsulating the address of a UCP worker.
 *
 * A UCP worker has a unique address that can is contained in a `ucp_address_t*` object,
 * this class encapsulates that object and provides methods to simplify its handling.
 */
class Address : public Component {
 private:
  ucp_address_t* _handle{nullptr};
  size_t _length{0};

  /**
   * @brief Private constructor of `ucxx::Address`.
   *
   * This is the internal implementation of `ucxx::Address` constructor, made private not
   * to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::createAddressFromWorker()`
   * - `ucxx::createAddressFromString()`
   * - `ucxx::Worker::getAddress()`
   *
   * @param[in] worker  the parent `set::shared_ptr<Worker>` component.
   * @param[in] address UCP address handle.
   * @param[in] length  length of the address byte-string in bytes.
   */
  Address(std::shared_ptr<Worker> worker, ucp_address_t* address, size_t length);

 public:
  Address()                          = delete;
  Address(const Address&)            = delete;
  Address& operator=(Address const&) = delete;
  Address(Address&& o)               = delete;
  Address& operator=(Address&& o)    = delete;

  ~Address();

  /**
   * @brief Constructor for `shared_ptr<ucxx::Address>` from worker.
   *
   * The constructor for a `shared_ptr<ucxx::Address>` object from a
   * `std::shared_ptr<ucxx::Worker>` to obtain its address.
   *
   * @param[in] worker  parent worker from which to get the address.
   *
   * @returns The `shared_ptr<ucxx::Address>` object.
   */
  friend std::shared_ptr<Address> createAddressFromWorker(std::shared_ptr<Worker> worker);

  /**
   * @brief Constructor for `shared_ptr<ucxx::Address>` from string.
   *
   * The constructor for a `shared_ptr<ucxx::Address>` object from the address extracted
   * as string from a remote `std::shared_ptr<ucxx::Worker>`.
   *
   * @param[in] addressString the string from which to create the address.
   *
   * @returns The `shared_ptr<ucxx::Address>` object.
   */
  friend std::shared_ptr<Address> createAddressFromString(std::string addressString);

  /**
   * @brief Get the underlying `ucp_address_t*` handle.
   *
   * Lifetime of the `ucp_address_t*` handle is managed by the `ucxx::Address` object and
   * its ownership is non-transferrable. Once the `ucxx::Address` is destroyed the handle
   * is not valid anymore, it is the user's responsibility to ensure the owner's lifetime
   * while using the handle.
   *
   * @code{.cpp}
   * // address is `std::shared_ptr<ucxx::Address>`
   * ucp_address_t* addressHandle = address->getHandle();
   * @endcode
   *
   * @returns The underlying `ucp_address_t` handle.
   */
  [[nodiscard]] ucp_address_t* getHandle() const;

  /**
   * @brief Get the length of the `ucp_address_t*` handle.
   *
   * Get the length of the `ucp_address_t*` handle, required to access the complete address
   * and prevent reading out-of-bound.
   *
   * @returns The length of the `ucp_address_t*` handle in bytes.
   */
  [[nodiscard]] size_t getLength() const;

  /**
   * @brief Get the address as a string.
   *
   * Convenience method to copy the underlying address to a `std::string` and return it as
   * a single object.
   *
   * @returns The underlying `ucp_address_t` handle.
   */
  [[nodiscard]] std::string getString() const;
};

}  // namespace ucxx
