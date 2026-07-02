/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include <ucxx/buffer.h>
#include <ucxx/component.h>
#include <ucxx/request_data.h>
#include <ucxx/typedefs.h>

namespace ucxx {

class Address;
class Context;
class Endpoint;
class Future;
class Listener;
class MemoryHandle;
class Notifier;
class RemoteKey;
class Request;
class RequestAmManaged;
class RequestAmSend;
class RequestAmRecvData;
class RequestEndpointClose;
class RequestFlush;
class RequestMem;
class RequestStream;
class RequestTag;
class RequestTagMulti;
class TagProbeInfo;
class Worker;

namespace detail {

[[nodiscard]] std::shared_ptr<Address> createAddressFromWorker(std::shared_ptr<Worker> worker);

[[nodiscard]] std::shared_ptr<Address> createAddressFromString(std::string_view addressString);

[[nodiscard]] std::shared_ptr<Listener> createListener(std::shared_ptr<Worker> worker,
                                                       uint16_t port,
                                                       ucp_listener_conn_callback_t callback,
                                                       void* callbackArgs);

[[nodiscard]] std::shared_ptr<RemoteKey> createRemoteKeyFromMemoryHandle(
  std::shared_ptr<MemoryHandle> memoryHandle);

[[nodiscard]] std::shared_ptr<RemoteKey> createRemoteKeyFromSerialized(
  std::shared_ptr<Endpoint> endpoint, SerializedRemoteKey serializedRemoteKey);

[[nodiscard]] std::shared_ptr<TagProbeInfo> createTagProbeInfo();

[[nodiscard]] std::shared_ptr<TagProbeInfo> createTagProbeInfo(const ucp_tag_recv_info_t& info,
                                                               ucp_tag_message_h handle);

}  // namespace detail

// Components
/**
 * @brief Constructor for `std::shared_ptr<ucxx::Address>` from a worker.
 */
[[nodiscard]] std::shared_ptr<Address> createAddressFromWorker(std::shared_ptr<Worker> worker);

/**
 * @brief Constructor for `std::shared_ptr<ucxx::Address>` from a string.
 */
[[nodiscard]] std::shared_ptr<Address> createAddressFromString(std::string_view addressString);
[[nodiscard]] std::shared_ptr<Context> createContext(const ConfigMap ucxConfig,
                                                     const uint64_t featureFlags);
[[nodiscard]] std::shared_ptr<Endpoint> createEndpointFromHostname(std::shared_ptr<Worker> worker,
                                                                   std::string ipAddress,
                                                                   uint16_t port,
                                                                   bool endpointErrorHandling);
[[nodiscard]] std::shared_ptr<Endpoint> createEndpointFromConnRequest(
  std::shared_ptr<Listener> listener, ucp_conn_request_h connRequest, bool endpointErrorHandling);
[[nodiscard]] std::shared_ptr<Endpoint> createEndpointFromWorkerAddress(
  std::shared_ptr<Worker> worker, std::shared_ptr<Address> address, bool endpointErrorHandling);

/**
 * @brief Constructor for `std::shared_ptr<ucxx::Listener>`.
 */
[[nodiscard]] std::shared_ptr<Listener> createListener(std::shared_ptr<Worker> worker,
                                                       uint16_t port,
                                                       ucp_listener_conn_callback_t callback,
                                                       void* callbackArgs);
[[nodiscard]] std::shared_ptr<Worker> createWorker(std::shared_ptr<Context> context,
                                                   const bool enableDelayedSubmission,
                                                   const bool enableFuture);
[[nodiscard]] std::shared_ptr<MemoryHandle> createMemoryHandle(
  std::shared_ptr<Context> context,
  const size_t size,
  void* buffer                       = nullptr,
  const ucs_memory_type_t memoryType = UCS_MEMORY_TYPE_HOST);

/**
 * @brief Constructor for `std::shared_ptr<ucxx::RemoteKey>` from a memory handle.
 */
[[nodiscard]] std::shared_ptr<RemoteKey> createRemoteKeyFromMemoryHandle(
  std::shared_ptr<MemoryHandle> memoryHandle);

/**
 * @brief Constructor for `std::shared_ptr<ucxx::RemoteKey>` from serialized data.
 */
[[nodiscard]] std::shared_ptr<RemoteKey> createRemoteKeyFromSerialized(
  std::shared_ptr<Endpoint> endpoint, SerializedRemoteKey serializedRemoteKey);

// Transfers
[[nodiscard]] std::shared_ptr<RequestAmManaged> createRequestAmManaged(
  std::shared_ptr<Endpoint> endpoint,
  const std::variant<data::AmSendManaged, data::AmReceiveManaged> requestData,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData);

[[deprecated(
  "Use createRequestAmManaged() instead.")]] [[nodiscard]] std::shared_ptr<RequestAmManaged>
createRequestAm(std::shared_ptr<Endpoint> endpoint,
                const std::variant<data::AmSendManaged, data::AmReceiveManaged> requestData,
                const bool enablePythonFuture,
                RequestCallbackUserFunction callbackFunction,
                RequestCallbackUserData callbackData);

[[nodiscard]] std::shared_ptr<RequestAmSend> createRequestAmSend(
  std::shared_ptr<Endpoint> endpoint,
  const data::AmSend& requestData,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData);

[[nodiscard]] std::shared_ptr<RequestAmRecvData> createRequestAmRecvData(
  std::shared_ptr<Worker> worker,
  const data::AmRecvData& requestData,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData);
[[nodiscard]] std::shared_ptr<RequestEndpointClose> createRequestEndpointClose(
  std::shared_ptr<Endpoint> endpoint,
  const data::EndpointClose requestData,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData);
[[nodiscard]] std::shared_ptr<RequestFlush> createRequestFlush(
  std::shared_ptr<Component> endpointOrWorker,
  const data::Flush requestData,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData);
[[nodiscard]] std::shared_ptr<RequestStream> createRequestStream(
  std::shared_ptr<Endpoint> endpoint,
  const std::variant<data::StreamSend, data::StreamReceive> requestData,
  const bool enablePythonFuture);
[[nodiscard]] std::shared_ptr<RequestTag> createRequestTag(
  std::shared_ptr<Component> endpointOrWorker,
  const std::variant<data::TagSend, data::TagReceive, data::TagReceiveWithHandle> requestData,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData);
[[nodiscard]] std::shared_ptr<RequestMem> createRequestMem(
  std::shared_ptr<Endpoint> endpoint,
  const std::variant<data::MemPut, data::MemGet> requestData,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData);
[[nodiscard]] std::shared_ptr<RequestTagMulti> createRequestTagMulti(
  std::shared_ptr<Endpoint> endpoint,
  const std::variant<data::TagMultiSend, data::TagMultiReceive> requestData,
  const bool enablePythonFuture);

/**
 * @brief Constructor for an unmatched `std::shared_ptr<ucxx::TagProbeInfo>`.
 */
[[nodiscard]] std::shared_ptr<TagProbeInfo> createTagProbeInfo();

/**
 * @brief Constructor for a matched `std::shared_ptr<ucxx::TagProbeInfo>`.
 */
[[nodiscard]] std::shared_ptr<TagProbeInfo> createTagProbeInfo(const ucp_tag_recv_info_t& info,
                                                               ucp_tag_message_h handle);

}  // namespace ucxx
