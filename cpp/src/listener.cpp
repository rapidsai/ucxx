/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <tuple>
#include <utility>

#include <ucp/api/ucp.h>

#include <internal/constructors.h>
#include <ucxx/exception.h>
#include <ucxx/listener.h>

#include <ucxx/utils/callback_notifier.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

Listener::Listener(std::shared_ptr<Worker> worker,
                   std::string host,
                   uint16_t port,
                   ucp_listener_conn_callback_t callback,
                   void* callbackArgs)
{
  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  _callback     = callback;
  _callbackArgs = callbackArgs;

  ucp_listener_params_t params = {
    .field_mask   = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER,
    .conn_handler = {.cb = connCallback, .arg = this}};
  auto info               = ucxx::utils::get_addrinfo(host.empty() ? nullptr : host.c_str(), port);
  params.sockaddr.addr    = info->ai_addr;
  params.sockaddr.addrlen = info->ai_addrlen;

  utils::ucsErrorThrow(ucp_listener_create(worker->getHandle(), &params, &_handle));
  ucxx_trace("ucxx::Listener created: %p, UCP handle: %p", this, _handle);

  ucp_listener_attr_t attr = {.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR};
  utils::ucsErrorThrow(ucp_listener_query(_handle, &attr));

  char ipString[INET6_ADDRSTRLEN];
  char portString[INET6_ADDRSTRLEN];
  ucxx::utils::sockaddr_get_ip_port_str(&attr.sockaddr, ipString, portString, INET6_ADDRSTRLEN);

  _ip   = std::string(ipString);
  _port = static_cast<uint16_t>(atoi(portString));

  setParent(worker);
}

namespace {

/**
 * @brief Number of listener callbacks the current thread is executing.
 *
 * Used to prevent `close()` from progressing the worker from within a listener
 * callback, which executes within `ucp_worker_progress()` that must not be nested.
 */
thread_local size_t listenerCallbackDepth{0};

struct ListenerCallbackScope {
  ListenerCallbackScope() { ++listenerCallbackDepth; }
  ~ListenerCallbackScope() { --listenerCallbackDepth; }
  ListenerCallbackScope(const ListenerCallbackScope&)            = delete;
  ListenerCallbackScope& operator=(const ListenerCallbackScope&) = delete;
  ListenerCallbackScope(ListenerCallbackScope&&)                 = delete;
  ListenerCallbackScope& operator=(ListenerCallbackScope&&)      = delete;
};

}  // namespace

Listener::~Listener() { close(); }

void Listener::connCallback(ucp_conn_request_h connRequest, void* arg)
{
  auto listener = reinterpret_cast<Listener*>(arg);

  if (listener->_closing.load()) {
    // Do not call the user callback, its argument may be released once `close()` returns.
    ucxx_debug(
      "ucxx::Listener %p is closing, rejecting connection request %p", listener, connRequest);
    listener->reject(connRequest);
    return;
  }

  ListenerCallbackScope scope{};
  listener->_callback(connRequest, listener->_callbackArgs);
}

void Listener::destroyHandle()
{
  std::lock_guard<std::mutex> lock(_handleMutex);
  if (_handle == nullptr) return;

  ucp_listener_destroy(_handle);
  ucxx_trace("ucxx::Listener closed: %p, UCP handle: %p", this, _handle);
  _handle = nullptr;
}

void Listener::close()
{
  // Reject connection requests delivered from now on, without calling the user callback.
  _closing.store(true);

  if (isClosed()) return;

  auto worker = std::static_pointer_cast<Worker>(_parent);

  if (worker->isProgressThreadRunning()) {
    // Every caller waits for the handle to be destroyed. `registerGenericPre` executes the
    // callback immediately if called from the progress thread (e.g., from within the
    // listener callback), so the progress thread never waits for itself. The handle
    // remains valid until destroyed, so that a listener callback that is executing
    // concurrently can still use it, therefore `_handleMutex` must not be held while
    // waiting.
    if (!worker->registerGenericPre([this]() { destroyHandle(); }, 10000000000 /* 10s */)) {
      ucxx_error(
        "ucxx::Listener %p timed out waiting for the progress thread to destroy the UCP "
        "listener %p, the listener callback argument must not be released",
        this,
        getHandle());
      return;
    }
    std::ignore = worker->registerGenericPost([]() {}, 10000000000 /* 10s */);
  } else {
    destroyHandle();
    // Progressing the worker from within the listener callback would nest
    // `ucp_worker_progress()`, which UCX does not allow.
    if (listenerCallbackDepth == 0) worker->progress();
  }
}

void Listener::reject(ucp_conn_request_h connRequest)
{
  std::lock_guard<std::mutex> lock(_handleMutex);
  if (_handle == nullptr) {
    ucxx_warn(
      "ucxx::Listener %p is closed, cannot reject connection request %p", this, connRequest);
    return;
  }

  auto status = ucp_listener_reject(_handle, connRequest);
  if (status != UCS_OK)
    ucxx_warn("ucxx::Listener %p failed to reject connection request %p: %s",
              this,
              connRequest,
              ucs_status_string(status));
}

bool Listener::isClosed()
{
  std::lock_guard<std::mutex> lock(_handleMutex);
  return _handle == nullptr;
}

namespace detail {

std::shared_ptr<Listener> ConstructorFactory::createListener(std::shared_ptr<Worker> worker,
                                                             std::string host,
                                                             uint16_t port,
                                                             ucp_listener_conn_callback_t callback,
                                                             void* callbackArgs)
{
  return std::shared_ptr<Listener>(
    new Listener(worker, std::move(host), port, callback, callbackArgs));
}

}  // namespace detail

EndpointBuilder Listener::endpointBuilder(ucp_conn_request_h connRequest)
{
  return EndpointBuilder(std::static_pointer_cast<Listener>(shared_from_this()), connRequest);
}

ucp_listener_h Listener::getHandle()
{
  std::lock_guard<std::mutex> lock(_handleMutex);
  return _handle;
}

uint16_t Listener::getPort() { return _port; }

std::string Listener::getIp() { return _ip; }

}  // namespace ucxx
