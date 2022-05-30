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

void _ucp_listener_destroy(ucp_listener_h ptr);

class Listener : public Component {
 private:
  std::unique_ptr<ucp_listener, void (*)(ucp_listener_h)> _handle{nullptr, _ucp_listener_destroy};
  std::string _ip{};
  uint16_t _port{0};

  Listener(std::shared_ptr<Worker> worker,
           uint16_t port,
           ucp_listener_conn_callback_t callback,
           void* callback_args);

 public:
  Listener()                = delete;
  Listener(const Listener&) = delete;
  Listener& operator=(Listener const&) = delete;
  Listener(Listener&& o)               = delete;
  Listener& operator=(Listener&& o) = delete;

  ~Listener();

  friend std::shared_ptr<Listener> createListener(std::shared_ptr<Worker> worker,
                                                  uint16_t port,
                                                  ucp_listener_conn_callback_t callback,
                                                  void* callback_args);

  std::shared_ptr<Endpoint> createEndpointFromConnRequest(ucp_conn_request_h conn_request,
                                                          bool endpoint_error_handling = true);

  ucp_listener_h get_handle();

  uint16_t getPort();
};

}  // namespace ucxx
