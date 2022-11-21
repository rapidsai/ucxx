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

void ucpListenerDestructor(ucp_listener_h ptr);

class Listener : public Component {
 private:
  std::unique_ptr<ucp_listener, void (*)(ucp_listener_h)> _handle{nullptr, ucpListenerDestructor};
  std::string _ip{};
  uint16_t _port{0};

  Listener(std::shared_ptr<Worker> worker,
           uint16_t port,
           ucp_listener_conn_callback_t callback,
           void* callbackArgs);

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
                                                  void* callbackArgs);

  std::shared_ptr<Endpoint> createEndpointFromConnRequest(ucp_conn_request_h connRequest,
                                                          bool endpointErrorHandling = true);

  ucp_listener_h getHandle();

  uint16_t getPort();

  std::string getIp();
};

}  // namespace ucxx
