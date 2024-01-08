/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ucxx/typedefs.h>

namespace ucxx {

class Address;
class Context;
class Endpoint;
class Future;
class Listener;
class Notifier;
class Request;
class RequestAm;
class RequestMem;
class RequestStream;
class RequestTag;
class RequestTagMulti;
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
                                     const bool enableFuture);

// Transfers
std::shared_ptr<RequestAm> createRequestAmSend(std::shared_ptr<Endpoint> endpoint,
                                               void* buffer,
                                               size_t length,
                                               ucs_memory_type_t memoryType,
                                               const bool enablePythonFuture,
                                               RequestCallbackUserFunction callbackFunction,
                                               RequestCallbackUserData callbackData);

std::shared_ptr<RequestAm> createRequestAmRecv(std::shared_ptr<Endpoint> endpoint,
                                               const bool enablePythonFuture,
                                               RequestCallbackUserFunction callbackFunction,
                                               RequestCallbackUserData callbackData);

std::shared_ptr<RequestStream> createRequestStream(std::shared_ptr<Endpoint> endpoint,
                                                   bool send,
                                                   void* buffer,
                                                   size_t length,
                                                   const bool enablePythonFuture);

std::shared_ptr<RequestMem> createRequestMem(std::shared_ptr<Endpoint> endpoint,
                                             bool send,
                                             void* buffer,
                                             size_t length,
                                             uint64_t remote_addr,
                                             ucp_rkey_h rkey,
                                             const bool enablePythonFuture,
                                             RequestCallbackUserFunction callbackFunction,
                                             RequestCallbackUserData callbackData);

std::shared_ptr<RequestTag> createRequestTag(std::shared_ptr<Component> endpointOrWorker,
                                             bool send,
                                             void* buffer,
                                             size_t length,
                                             ucp_tag_t tag,
                                             const bool enablePythonFuture,
                                             RequestCallbackUserFunction callbackFunction,
                                             RequestCallbackUserData callbackData);

std::shared_ptr<RequestTagMulti> createRequestTagMultiSend(std::shared_ptr<Endpoint> endpoint,
                                                           const std::vector<void*>& buffer,
                                                           const std::vector<size_t>& size,
                                                           const std::vector<int>& isCUDA,
                                                           const ucp_tag_t tag,
                                                           const bool enablePythonFuture);

std::shared_ptr<RequestTagMulti> createRequestTagMultiRecv(std::shared_ptr<Endpoint> endpoint,
                                                           const ucp_tag_t tag,
                                                           const bool enablePythonFuture);

}  // namespace ucxx
