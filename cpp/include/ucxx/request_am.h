/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <string>
#include <utility>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>

namespace ucxx {

class Buffer;

namespace internal {
class RecvAmMessage;
}  // namespace internal

class RequestAm : public Request {
 private:
  friend class internal::RecvAmMessage;

  /**
   * @brief Private constructor of `ucxx::RequestAm`.
   *
   * This is the internal implementation of `ucxx::RequestAm` constructor, made private
   * not to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Endpoint::amSend()`
   * - `ucxx::createRequestAmSend()`
   * - `ucxx::Endpoint::amReceive()`
   * - `ucxx::createRequestAmReceive()`
   *
   * @throws ucxx::Error  if `endpoint` is not a valid `std::shared_ptr<ucxx::Endpoint>`.
   *
   * @param[in] endpoint            the parent endpoint.
   * @param[in] requestData         container of the specified message type, including all
   *                                type-specific data.
   * @param[in] operationName       a human-readable operation name to help identifying
   *                                requests by their types when UCXX logging is enabled.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   */
  RequestAm(std::shared_ptr<Component> endpointOrWorker,
            const data::RequestData requestData,
            const std::string operationName,
            const bool enablePythonFuture                = false,
            RequestCallbackUserFunction callbackFunction = nullptr,
            RequestCallbackUserData callbackData         = nullptr);

 public:
  /**
   * @brief Constructor for `std::shared_ptr<ucxx::RequestAm>`.
   *
   * The constructor for a `std::shared_ptr<ucxx::RequestAm>` object, creating an active
   * message request, returning a pointer to a request object that can be later awaited and
   * checked for errors. This is a non-blocking operation, and the status of the transfer
   * must be verified from the resulting request object before the data can be released if
   * this is a send operation, or consumed if this is a receive operation. Received data is
   * available via the `getRecvBuffer()` method if the receive transfer request completed
   * successfully.
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
   * @returns The `shared_ptr<ucxx::RequestAm>` object
   */
  friend std::shared_ptr<RequestAm> createRequestAm(std::shared_ptr<Endpoint> endpoint,
                                                    const data::RequestData requestData,
                                                    const bool enablePythonFuture,
                                                    RequestCallbackUserFunction callbackFunction,
                                                    RequestCallbackUserData callbackData);

  virtual void populateDelayedSubmission();

  /**
   * @brief Create and submit an active message send request.
   *
   * This is the method that should be called to actually submit an active message send
   * request. It is meant to be called from `populateDelayedSubmission()`, which is decided
   * at the discretion of `std::shared_ptr<ucxx::Worker>`. See `populateDelayedSubmission()`
   * for more details.
   */
  void request();

  /**
   * @brief Receive callback registered by `ucxx::Worker`.
   *
   * This is the receive callback registered by the `ucxx::Worker` to handle incoming active
   * messages. For each incoming active message, a proper buffer will be allocated based on
   * the header sent by the remote endpoint using the default allocator or one registered by
   * the user via `ucxx::Worker::registerAmAllocator()`. Following that, the message is
   * immediately received onto the buffer and a `UCS_OK` or the proper error status is set
   * onto the `RequestAm` that is returned to the user, or will be later handled by another
   * callback when the message is ready. If the callback is executed when a user has already
   * requested received of the active message, the buffer and status will be set on the
   * earliest request, otherwise a new request is created and saved in a pool that will be
   * already populated and ready for consumption or waiting for the internal callback when
   * requested.
   *
   * This is always called by `ucp_worker_progress()`, and thus will happen in the same
   * thread that is called from, when using the worker progress thread, this is called from
   * the progress thread.
   *
   * param[in,out] arg  pointer to the `AmData` object held by the `ucxx::Worker` who
   *                    registered this callback.
   * param[in] header pointer to the header containing the sender buffer's memory type.
   * param[in] header_length  length in bytes of the receive header.
   * param[in] data pointer to the buffer containing the remote endpoint's send data.
   * param[in] length the length in bytes of the message to be received.
   * param[in] param  UCP parameters of the active message being received.
   */
  static ucs_status_t recvCallback(void* arg,
                                   const void* header,
                                   size_t header_length,
                                   void* data,
                                   size_t length,
                                   const ucp_am_recv_param_t* param);

  std::shared_ptr<Buffer> getRecvBuffer() override;
};

}  // namespace ucxx
