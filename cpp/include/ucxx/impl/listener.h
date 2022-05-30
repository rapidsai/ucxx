/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

void ucpListenerDestructor(ucp_listener_h ptr)
{
  if (ptr != nullptr) ucp_listener_destroy(ptr);
}

Listener::Listener(std::shared_ptr<Worker> worker,
                   uint16_t port,
                   ucp_listener_conn_callback_t callback,
                   void* callbackArgs)
{
  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  ucp_listener_params_t params;
  ucp_listener_attr_t attr;

  params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
  params.conn_handler.cb  = callback;
  params.conn_handler.arg = callbackArgs;

  if (ucxx::utils::sockaddr_set(&params.sockaddr, NULL, port))
    // throw std::bad_alloc("Failed allocation of sockaddr")
    throw std::bad_alloc();
  std::unique_ptr<ucs_sock_addr_t, void (*)(ucs_sock_addr_t*)> sockaddr(&params.sockaddr,
                                                                        ucxx::utils::sockaddr_free);

  ucp_listener_h handle = nullptr;
  utils::assert_ucs_status(ucp_listener_create(worker->getHandle(), &params, &handle));
  _handle = std::unique_ptr<ucp_listener, void (*)(ucp_listener_h)>(handle, ucpListenerDestructor);

  attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
  utils::assert_ucs_status(ucp_listener_query(_handle.get(), &attr));

  size_t MAX_STR_LEN = 50;
  char ipString[MAX_STR_LEN];
  char portString[MAX_STR_LEN];
  ucxx::utils::sockaddr_get_ip_port_str(&attr.sockaddr, ipString, portString, MAX_STR_LEN);

  _ip   = std::string(ipString);
  _port = (uint16_t)atoi(portString);

  setParent(worker);
}

Listener::~Listener() {}

std::shared_ptr<Listener> createListener(std::shared_ptr<Worker> worker,
                                         uint16_t port,
                                         ucp_listener_conn_callback_t callback,
                                         void* callbackArgs)
{
  return std::shared_ptr<Listener>(new Listener(worker, port, callback, callbackArgs));
}

std::shared_ptr<Endpoint> Listener::createEndpointFromConnRequest(ucp_conn_request_h connRequest,
                                                                  bool endpointErrorHandling)
{
  auto listener = std::dynamic_pointer_cast<Listener>(shared_from_this());
  auto endpoint = ucxx::createEndpointFromConnRequest(listener, connRequest, endpointErrorHandling);
  return endpoint;
}

ucp_listener_h Listener::getHandle() { return _handle.get(); }

uint16_t Listener::getPort() { return _port; }

}  // namespace ucxx
