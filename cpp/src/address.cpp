/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <ucxx/address.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

Address::Address(ucp_address_t* address, size_t length, std::shared_ptr<ucxx::Worker> worker)
  : _handle{(char*)address}, _length{length}
{
  if (worker != nullptr) setParent(worker);
}

Address::~Address()
{
  if (_handle == nullptr) return;

  auto worker = std::dynamic_pointer_cast<Worker>(getParent());
  if (worker == nullptr) {
    delete[] _handle;
  } else {
    ucp_worker_release_address(worker->getHandle(), (ucp_address_t*)_handle);
  }
}

std::shared_ptr<Address> createAddressFromWorker(std::shared_ptr<ucxx::Worker> worker)
{
  ucp_worker_h ucp_worker = worker->getHandle();
  ucp_address_t* address{nullptr};
  size_t length = 0;

  utils::assert_ucs_status(ucp_worker_get_address(ucp_worker, &address, &length));
  return std::shared_ptr<Address>(new Address(address, length, worker));
}

std::shared_ptr<Address> createAddressFromString(std::string addressString)
{
  char* address = new char[addressString.length()];
  size_t length = addressString.length();
  memcpy((char*)address, addressString.c_str(), length);
  return std::shared_ptr<Address>(new Address((ucp_address_t*)address, length, nullptr));
}

ucp_address_t* Address::getHandle() const { return (ucp_address_t*)_handle; }

size_t Address::getLength() const { return _length; }

std::string Address::getString() const { return std::string{(char*)_handle, _length}; }

}  // namespace ucxx
