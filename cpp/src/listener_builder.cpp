/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <string>
#include <utility>

#include <ucxx/constructors.h>
#include <ucxx/detail/builder_utils.h>
#include <ucxx/listener_builder.h>

namespace ucxx {

struct ListenerBuilder::Impl {
  std::shared_ptr<Worker> worker{nullptr};
  std::string ipAddress{};
  uint16_t port{0};
  ucp_listener_conn_callback_t callback{nullptr};
  void* callbackArgs{nullptr};

  Impl(std::shared_ptr<Worker> w, uint16_t p, ucp_listener_conn_callback_t cb, void* cbArgs)
    : worker(std::move(w)), port(p), callback(cb), callbackArgs(cbArgs)
  {
  }
};

ListenerBuilder::ListenerBuilder(std::shared_ptr<Worker> worker,
                                 uint16_t port,
                                 ucp_listener_conn_callback_t callback,
                                 void* callbackArgs)
  : _impl(std::make_unique<Impl>(std::move(worker), port, callback, callbackArgs))
{
}

UCXX_BUILDER_PIMPL_DEFAULTS(ListenerBuilder, Listener)

ListenerBuilder& ListenerBuilder::ipAddress(std::string ipAddress)
{
  _impl->ipAddress = std::move(ipAddress);
  return *this;
}

std::shared_ptr<Listener> ListenerBuilder::build()
{
  return detail::createListener(
    _impl->worker, _impl->ipAddress, _impl->port, _impl->callback, _impl->callbackArgs);
}

}  // namespace ucxx
