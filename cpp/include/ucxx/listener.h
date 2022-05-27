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

class UCXXListener : public UCXXComponent {
 private:
  std::unique_ptr<ucp_listener, void (*)(ucp_listener_h)> _handle{nullptr, _ucp_listener_destroy};
  std::string _ip{};
  uint16_t _port{0};

  UCXXListener(std::shared_ptr<UCXXWorker> worker,
               uint16_t port,
               ucp_listener_conn_callback_t callback,
               void* callback_args);

 public:
  UCXXListener()                    = delete;
  UCXXListener(const UCXXListener&) = delete;
  UCXXListener& operator=(UCXXListener const&) = delete;
  UCXXListener(UCXXListener&& o)               = delete;
  UCXXListener& operator=(UCXXListener&& o) = delete;

  ~UCXXListener();

  friend std::shared_ptr<UCXXListener> createListener(std::shared_ptr<UCXXWorker> worker,
                                                      uint16_t port,
                                                      ucp_listener_conn_callback_t callback,
                                                      void* callback_args);

  std::shared_ptr<UCXXEndpoint> createEndpointFromConnRequest(ucp_conn_request_h conn_request,
                                                              bool endpoint_error_handling = true);

  ucp_listener_h get_handle();

  uint16_t getPort();
};

}  // namespace ucxx
