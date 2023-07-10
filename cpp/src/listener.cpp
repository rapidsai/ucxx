/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <condition_variable>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <ucp/api/ucp.h>

#include <ucxx/exception.h>
#include <ucxx/listener.h>
#include <ucxx/utils/condition.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>

#include <iostream>

namespace ucxx {

Listener::Listener(std::shared_ptr<Worker> worker,
                   uint16_t port,
                   ucp_listener_conn_callback_t callback,
                   void* callbackArgs)
{
  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  ucp_listener_params_t params = {
    .field_mask   = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER,
    .conn_handler = {.cb = callback, .arg = callbackArgs}};
  auto info               = ucxx::utils::get_addrinfo(NULL, port);
  params.sockaddr.addr    = info->ai_addr;
  params.sockaddr.addrlen = info->ai_addrlen;

  utils::ucsErrorThrow(ucp_listener_create(worker->getHandle(), &params, &_handle));
  ucxx_trace("Listener created: %p", _handle);

  ucp_listener_attr_t attr = {.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR};
  utils::ucsErrorThrow(ucp_listener_query(_handle, &attr));

  char ipString[INET6_ADDRSTRLEN];
  char portString[INET6_ADDRSTRLEN];
  ucxx::utils::sockaddr_get_ip_port_str(&attr.sockaddr, ipString, portString, INET6_ADDRSTRLEN);

  _ip   = std::string(ipString);
  _port = (uint16_t)atoi(portString);

  setParent(worker);
}

Listener::~Listener()
{
  auto worker = std::static_pointer_cast<Worker>(_parent);

  if (worker->isProgressThreadRunning()) {
    auto statusMutex             = std::make_shared<std::mutex>();
    auto statusConditionVariable = std::make_shared<std::condition_variable>();
    auto pre                     = std::make_shared<bool>(false);
    auto post                    = std::make_shared<bool>(false);

    auto setterPre = [this, pre]() {
      ucp_listener_destroy(_handle);
      *pre = true;
    };
    auto getterPre = [pre]() { return *pre; };

    worker->registerGenericPre([&statusMutex, &statusConditionVariable, &setterPre]() {
      ucxx::utils::conditionSetter(statusMutex, statusConditionVariable, setterPre);
    });
    ucxx::utils::conditionGetter(statusMutex, statusConditionVariable, pre, getterPre);

    auto setterPost = [this, post]() { *post = true; };
    auto getterPost = [post]() { return *post; };
    worker->registerGenericPost([&statusMutex, &statusConditionVariable, &setterPost]() {
      ucxx::utils::conditionSetter(statusMutex, statusConditionVariable, setterPost);
    });
    ucxx::utils::conditionGetter(statusMutex, statusConditionVariable, post, getterPost);
  } else {
    ucp_listener_destroy(this->_handle);
    worker->progress();
  }

  ucxx_trace("Listener destroyed: %p", this->_handle);
}

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

ucp_listener_h Listener::getHandle() { return _handle; }

uint16_t Listener::getPort() { return _port; }

std::string Listener::getIp() { return _ip; }

}  // namespace ucxx
