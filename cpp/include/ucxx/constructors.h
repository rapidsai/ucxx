/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>

#include <ucxx/typedefs.h>

#if UCXX_ENABLE_PYTHON
#include <ucxx/buffer.h>
#endif

namespace ucxx {

class Address;
class RequestTagMulti;
class Context;
class Endpoint;
class Listener;
class Request;
class RequestStream;
class RequestTag;
class Worker;

// Components
std::shared_ptr<Address> createAddressFromWorker(std::shared_ptr<ucxx::Worker> worker);

std::shared_ptr<Address> createAddressFromString(std::string addressString);

std::shared_ptr<Context> createContext(const ConfigMap ucxConfig, const uint64_t featureFlags);

std::shared_ptr<Endpoint> createEndpointFromHostname(std::shared_ptr<Worker> worker,
                                                     std::string ipAddress,
                                                     uint16_t port,
                                                     bool endpointErrorHandling);

std::shared_ptr<Endpoint> createEndpointFromConnRequest(std::shared_ptr<Listener> listener,
                                                        ucp_conn_request_h connRequest,
                                                        bool endpointErrorHandling);

std::shared_ptr<Endpoint> createEndpointFromWorkerAddress(std::shared_ptr<Worker> worker,
                                                          std::shared_ptr<Address> address,
                                                          bool endpointErrorHandling);

std::shared_ptr<Listener> createListener(std::shared_ptr<Worker> worker,
                                         uint16_t port,
                                         ucp_listener_conn_callback_t callback,
                                         void* callback_args);

std::shared_ptr<Worker> createWorker(std::shared_ptr<Context> context,
                                     const bool enableDelayedSubmission,
                                     const bool enablePythonFuture);

// Transfers
std::shared_ptr<RequestStream> createRequestStream(std::shared_ptr<Endpoint> endpoint,
                                                   bool send,
                                                   void* buffer,
                                                   size_t length,
                                                   const bool enablePythonFuture);

std::shared_ptr<RequestTag> createRequestTag(
  std::shared_ptr<Component> endpointOrWorker,
  bool send,
  void* buffer,
  size_t length,
  ucp_tag_t tag,
  const bool enablePythonFuture,
  std::function<void(std::shared_ptr<void>)> callbackFunction,
  std::shared_ptr<void> callbackData);

std::shared_ptr<RequestTagMulti> createRequestTagMultiSend(std::shared_ptr<Endpoint> endpoint,
                                                           std::vector<void*>& buffer,
                                                           std::vector<size_t>& size,
                                                           std::vector<int>& isCUDA,
                                                           const ucp_tag_t tag,
                                                           const bool enablePythonFuture);

std::shared_ptr<RequestTagMulti> createRequestTagMultiRecv(std::shared_ptr<Endpoint> endpoint,
                                                           const ucp_tag_t tag,
                                                           const bool enablePythonFuture);

#if UCXX_ENABLE_PYTHON
namespace python {

class Notifier;

std::shared_ptr<Notifier> createNotifier();

}  // namespace python
#endif

}  // namespace ucxx
