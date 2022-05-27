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

class UCXXAddress : public UCXXComponent {
 private:
  char* _handle{nullptr};
  size_t _length{0};

  UCXXAddress(ucp_address_t* address, size_t length, std::shared_ptr<ucxx::UCXXWorker> worker);

 public:
  UCXXAddress()                   = delete;
  UCXXAddress(const UCXXAddress&) = delete;
  UCXXAddress& operator=(UCXXAddress const&) = delete;
  UCXXAddress(UCXXAddress&& o)               = delete;
  UCXXAddress& operator=(UCXXAddress&& o) = delete;

  ~UCXXAddress();

  friend std::shared_ptr<UCXXAddress> createAddressFromWorker(
    std::shared_ptr<ucxx::UCXXWorker> worker);

  friend std::shared_ptr<UCXXAddress> createAddressFromString(std::string addressString);

  ucp_address_t* getHandle() const;

  size_t getLength() const;

  std::string getString() const;
};

}  // namespace ucxx
