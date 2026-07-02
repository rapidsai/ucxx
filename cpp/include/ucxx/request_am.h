/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request.h>
#include <ucxx/request_am_builder.h>
#include <ucxx/typedefs.h>

namespace ucxx {

class Buffer;

namespace internal {
class ManagedRecvAmMessage;
}  // namespace internal

/**
 * @brief Request type for UCXX managed Active Message operations.
 *
 * Managed Active Messages are UCXX-managed transfers built on UCX Active Messages. Sends use
 * UCXX's managed AM handler on the remote worker and carry a UCXX header containing the
 * sender memory type, receiver allocation policy, optional receiver callback information, and
 * optional user header bytes.
 *
 * A `RequestAmManaged` may represent either a send request or a receive request. Receive
 * requests are completed by the worker's managed AM receive callback, which deserializes the
 * UCXX header, allocates the receive buffer, handles eager or rendezvous payload delivery,
 * and either matches the message with an `amManagedRecv()` request or invokes a registered
 * receiver callback.
 *
 * For application-defined AM IDs, application-defined wire headers, or direct UCX AM receive
 * callback handling, use `Endpoint::amSend()` with `Worker::setAmHandler()`.
 */
class RequestAmManaged : public Request {
 private:
  friend class internal::ManagedRecvAmMessage;

  std::vector<std::byte> _header{};  ///< Retain copy of header bytes for send requests as
                                     ///< workaround for
                                     ///< https://github.com/openucx/ucx/issues/10424

  /**
   * @brief Private constructor of `ucxx::RequestAmManaged`.
   *
   * This constructor creates the internal request object for either a managed AM send or a
   * managed AM receive, depending on `requestData`. It is private to ensure requests are
   * owned through `std::shared_ptr` and participate in UCXX request lifetime management.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Endpoint::amManagedSend()`
   * - `ucxx::createRequestAmManaged()`
   * - `ucxx::Endpoint::amManagedRecv()`
   * - `ucxx::Endpoint::amSendBuilder()`
   * - `ucxx::Endpoint::amRecvBuilder()`
   * - `ucxx::requestAmBuilder()`
   *
   * @throws ucxx::Error  if `endpoint` is not a valid `std::shared_ptr<ucxx::Endpoint>`.
   *
   * @param[in] endpointOrWorker    the parent endpoint or worker.
   * @param[in] requestData         container of the specified message type, including all
   *                                type-specific data.
   * @param[in] operationName       a human-readable operation name to help identifying
   *                                requests by their types when UCXX logging is enabled.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   */
  RequestAmManaged(std::shared_ptr<Component> endpointOrWorker,
                   const std::variant<data::AmSendManaged, data::AmReceiveManaged> requestData,
                   std::string operationName,
                   const bool enablePythonFuture                = false,
                   RequestCallbackUserFunction callbackFunction = nullptr,
                   RequestCallbackUserData callbackData         = nullptr);

 public:
  /**
   * @brief Factory for a managed Active Message request.
   *
   * Creates a managed AM send request when `requestData` contains `data::AmSendManaged`, or
   * a managed AM receive request when it contains `data::AmReceiveManaged`. Send requests are
   * later submitted with `RequestAmManaged::request()`. Receive requests are completed by the
   * worker's managed AM receive callback and may match a message that arrived before or after
   * the receive request was created.
   *
   * Received payload data is available via `getRecvBuffer()` after a receive request
   * completes successfully. Sender-provided user header bytes are available via
   * `getRecvHeader()`.
   *
   * @note If a `callbackFunction` is specified, the lifetime of `callbackData` and of any
   * other objects used in the scope of `callbackFunction` must be guaranteed by the caller
   * until it executes or `isCompleted()` becomes true. The `callbackFunction` executes in
   * the thread progressing the `ucxx::Worker`, unless the request completes immediately,
   * in which case the callback will also execute immediately within the calling thread and
   * before the method returns.
   *
   * @throws ucxx::Error  if `endpoint` is not a valid
   *                      `std::shared_ptr<ucxx::Endpoint>`.
   *
   * @param[in] endpoint            the parent endpoint.
   * @param[in] requestData         container of the specified message type, including all
   *                                type-specific data.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   *
   * @returns The `shared_ptr<ucxx::RequestAmManaged>` object
   */
  friend std::shared_ptr<RequestAmManaged> createRequestAmManaged(
    std::shared_ptr<Endpoint> endpoint,
    const std::variant<data::AmSendManaged, data::AmReceiveManaged> requestData,
    const bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);

  /**
   * @brief Deprecated factory for `RequestAmManaged`.
   * @deprecated Use `createRequestAmManaged()` instead.
   */
  friend std::shared_ptr<RequestAmManaged> createRequestAm(
    std::shared_ptr<Endpoint> endpoint,
    const std::variant<data::AmSendManaged, data::AmReceiveManaged> requestData,
    const bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);

  /**
   * @brief Cancel the request.
   *
   * Cancel the request. Often called by the error handler or parent's object
   * destructor but may be called by the user to cancel the request as well.
   */
  void cancel() override;

  void populateDelayedSubmission() override;

  /**
   * @brief Create and submit a managed active message send request.
   *
   * Serializes the managed AM header, calls `ucp_am_send_nbx` on the reserved managed AM
   * handler ID, and publishes the resulting UCX request. This method is valid only for
   * send-side `RequestAmManaged` objects and is normally called by
   * `populateDelayedSubmission()`.
   */
  void request();

  /**
   * @brief Receive callback registered by `ucxx::Worker` for the managed AM API.
   *
   * Handles incoming managed Active Messages for the worker-managed AM handler. The callback
   * deserializes the UCXX header, identifies the remote endpoint, allocates a receive buffer
   * according to the sender memory type and memory policy, and completes payload transfer for
   * either eager or rendezvous messages.
   *
   * Messages without receiver callback information are matched through the worker's
   * managed-receive pools, allowing either request-before-message or message-before-request
   * ordering. Messages with receiver callback information are delivered to the registered
   * callback after the receive request completes.
   *
   * @param[in,out] arg            pointer to the `ManagedAmData` object held by the worker.
   * @param[in]     header         header containing serialized UCXX metadata.
   * @param[in]     header_length  length in bytes of the receive header.
   * @param[in]     data           eager payload pointer or rendezvous data descriptor.
   * @param[in]     length         length in bytes of the message payload.
   * @param[in]     param          UCP parameters of the active message being received.
   */
  [[nodiscard]] static ucs_status_t recvCallback(void* arg,
                                                 const void* header,
                                                 size_t header_length,
                                                 void* data,
                                                 size_t length,
                                                 const ucp_am_recv_param_t* param);

  [[nodiscard]] std::shared_ptr<Buffer> getRecvBuffer() override;

  [[nodiscard]] std::string getRecvHeader() override;
};

/**
 * @brief Deprecated alias for `RequestAmManaged`.
 * @deprecated Use `RequestAmManaged` directly.
 */
[[deprecated("Use RequestAmManaged instead.")]] typedef RequestAmManaged RequestAm;

}  // namespace ucxx
