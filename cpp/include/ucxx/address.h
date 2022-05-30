/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/worker.h>

namespace ucxx {

class Address : public Component {
 private:
  char* _handle{nullptr};
  size_t _length{0};

  Address(ucp_address_t* address, size_t length, std::shared_ptr<ucxx::Worker> worker);

 public:
  Address()               = delete;
  Address(const Address&) = delete;
  Address& operator=(Address const&) = delete;
  Address(Address&& o)               = delete;
  Address& operator=(Address&& o) = delete;

  ~Address();

  friend std::shared_ptr<Address> createAddressFromWorker(std::shared_ptr<ucxx::Worker> worker);

  friend std::shared_ptr<Address> createAddressFromString(std::string addressString);

  ucp_address_t* getHandle() const;

  size_t getLength() const;

  std::string getString() const;
};

}  // namespace ucxx
