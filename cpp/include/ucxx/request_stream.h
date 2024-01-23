/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request.h>
#include <ucxx/request_data.h>
#include <ucxx/typedefs.h>

namespace ucxx {

/**
 * @brief Send or receive a message with the UCX Stream API.
 *
 * Send or receive a message with the UCX Stream API, using non-blocking UCP calls
 * `ucp_stream_send_nbx` or `ucp_stream_recv_nbx`.
 */
class RequestStream : public Request {
 private:
  /**
   * @brief Private constructor of `ucxx::RequestStream`.
   *
   * This is the internal implementation of `ucxx::RequestStream` constructor, made private
   * not to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Endpoint::streamRecv()`
   * - `ucxx::Endpoint::streamSend()`
   * - `ucxx::createRequestStream()`
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component
   * @param[in] requestData         container of the specified message type, including all
   *                                type-specific data.
   * @param[in] operationName       a human-readable operation name to help identifying
   *                                requests by their types when UCXX logging is enabled.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   */
  RequestStream(std::shared_ptr<Endpoint> endpoint,
                const std::variant<data::StreamSend, data::StreamReceive> requestData,
                const std::string operationName,
                const bool enablePythonFuture = false);

 public:
  /**
   * @brief Constructor for `std::shared_ptr<ucxx::RequestStream>`.
   *
   * The constructor for a `std::shared_ptr<ucxx::RequestStream>` object, creating a send
   * or receive stream request, returning a pointer to a request object that can be later
   * awaited and checked for errors. This is a non-blocking operation, and the status of
   * the transfer must be verified from the resulting request object before the data can be
   * released (for a send operation) or consumed (for a receive operation).
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component
   * @param[in] requestData         container of the specified message type, including all
   *                                type-specific data.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   *
   * @returns The `shared_ptr<ucxx::RequestStream>` object
   */
  friend std::shared_ptr<RequestStream> createRequestStream(
    std::shared_ptr<Endpoint> endpoint,
    const std::variant<data::StreamSend, data::StreamReceive> requestData,
    const bool enablePythonFuture);

  virtual void populateDelayedSubmission();

  /**
   * @brief Create and submit a stream request.
   *
   * This is the method that should be called to actually submit a stream request. It is
   * meant to be called from `populateDelayedSubmission()`, which is decided at the
   * discretion of `std::shared_ptr<ucxx::Worker>`. See `populateDelayedSubmission()` for
   * more details.
   */
  void request();

  /**
   * @brief Implementation of the stream receive request callback.
   *
   * Implementation of the stream receive request callback. Verify whether the message was
   * truncated and set that state if necessary, and finally dispatch
   * `ucxx::Request::callback()`.
   *
   * WARNING: This is not intended to be called by the user, but it currently needs to be
   * a public method so that UCX may access it. In future changes this will be moved to
   * an internal object and remove this method from the public API.
   *
   * @param[in] request the UCX request pointer.
   * @param[in] status  the completion status of the request.
   * @param[in] length  length of message received used to verify for truncation.
   */
  void callback(void* request, ucs_status_t status, size_t length);

  /**
   * @brief Callback executed by UCX when a stream send request is completed.
   *
   * Callback executed by UCX when a stream send request is completed, that will dispatch
   * `ucxx::Request::callback()`.
   *
   * WARNING: This is not intended to be called by the user, but it currently needs to be
   * a public method so that UCX may access it. In future changes this will be moved to
   * an internal object and remove this method from the public API.
   *
   * @param[in] request the UCX request pointer.
   * @param[in] status  the completion status of the request.
   * @param[in] arg     the pointer to the `ucxx::Request` object that created the
   *                    transfer, effectively `this` pointer as seen by `request()`.
   */
  static void streamSendCallback(void* request, ucs_status_t status, void* arg);

  /**
   * @brief Callback executed by UCX when a stream receive request is completed.
   *
   * Callback executed by UCX when a stream receive request is completed, that will
   * dispatch `ucxx::RequestStream::callback()`.
   *
   * WARNING: This is not intended to be called by the user, but it currently needs to be
   * a public method so that UCX may access it. In future changes this will be moved to
   * an internal object and remove this method from the public API.
   *
   * @param[in] request the UCX request pointer.
   * @param[in] status  the completion status of the request.
   * @param[in] length  length of message received used to verify for truncation.
   * @param[in] arg     the pointer to the `ucxx::Request` object that created the
   *                    transfer, effectively `this` pointer as seen by `request()`.
   */
  static void streamRecvCallback(void* request, ucs_status_t status, size_t length, void* arg);
};

}  // namespace ucxx
