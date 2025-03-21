/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <string_view>

#include <ucxx/address.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

Address::Address(std::shared_ptr<Worker> worker, ucp_address_t* address, size_t length)
  : _handle{address}, _length{length}
{
  if (worker != nullptr) setParent(worker);
}

Address::~Address()
{
  if (_handle == nullptr) return;

  auto worker = std::dynamic_pointer_cast<Worker>(getParent());
  if (worker == nullptr) {
    delete[] reinterpret_cast<char*>(_handle);
  } else {
    ucp_worker_release_address(worker->getHandle(), _handle);
  }
}

std::shared_ptr<Address> createAddressFromWorker(std::shared_ptr<Worker> worker)
{
  ucp_worker_h ucp_worker = worker->getHandle();
  ucp_address_t* address{nullptr};
  size_t length = 0;

  utils::ucsErrorThrow(ucp_worker_get_address(ucp_worker, &address, &length));
  return std::shared_ptr<Address>(new Address(worker, address, length));
}

std::shared_ptr<Address> createAddressFromString(std::string_view addressString)
{
  ucp_address_t* address = reinterpret_cast<ucp_address_t*>(new char[addressString.length()]);
  size_t length          = addressString.length();
  memcpy(address, addressString.data(), length);
  return std::shared_ptr<Address>(new Address(nullptr, address, length));
}

ucp_address_t* Address::getHandle() const { return _handle; }

size_t Address::getLength() const { return _length; }

std::string_view Address::getString() const
{
  return std::string_view{reinterpret_cast<const char*>(_handle), _length};
}

}  // namespace ucxx
