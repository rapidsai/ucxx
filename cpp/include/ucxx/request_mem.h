/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <utility>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>

namespace ucxx {

class Buffer;

namespace internal {
class RecvMemMessage;
}  // namespace internal

class RequestMem : public Request {
 private:
  friend class internal::RecvMemMessage;

  uint64_t _remote_addr{};
  ucp_rkey_h _rkey{};

  /**
   * @brief Private constructor of `ucxx::RequestAm` send.
   *
   * This is the internal implementation of `ucxx::RequestAm` send constructor, made private
   * not to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Endpoint::amSend()`
   * - `ucxx::createRequestAmSend()`
   *
   * @throws ucxx::Error  if `endpoint` is not a valid `std::shared_ptr<ucxx::Endpoint>`.
   *
   * @param[in] endpoint            the parent endpoint.
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the active message to be sent.
   * @param[in] memoryType          the memory type of the buffer.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   */
  RequestMem(std::shared_ptr<Endpoint> endpoint,
             bool send,
             void* buffer,
             size_t length,
             uint64_t remote_addr,
             ucp_rkey_h rkey,
             const bool enablePythonFuture                = false,
             RequestCallbackUserFunction callbackFunction = nullptr,
             RequestCallbackUserData callbackData         = nullptr);

 public:
  /**
   * @brief Constructor for `std::shared_ptr<ucxx::RequestAm>` send.
   *
   * The constructor for a `std::shared_ptr<ucxx::RequestAm>` object, creating a send active
   * message request, returning a pointer to a request object that can be later awaited and
   * checked for errors. This is a non-blocking operation, and the status of the transfer
   * must be verified from the resulting request object before the data can be
   * released.
   *
   * @throws ucxx::Error  if `endpoint` is not a valid
   *                      `std::shared_ptr<ucxx::Endpoint>`.
   *
   * @param[in] endpoint            the parent endpoint.
   * @param[in] buffer              a raw pointer to the data to be transferred.
   * @param[in] length              the size in bytes of the tag message to be transferred.
   * @param[in] memoryType          the memory type of the buffer.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   *
   * @returns The `shared_ptr<ucxx::RequestAm>` object
   */
  friend std::shared_ptr<RequestMem> createRequestMem(std::shared_ptr<Endpoint> endpoint,
                                                      bool send,
                                                      void* buffer,
                                                      size_t length,
                                                      uint64_t remote_addr,
                                                      ucp_rkey_h rkey,
                                                      const bool enablePythonFuture,
                                                      RequestCallbackUserFunction callbackFunction,
                                                      RequestCallbackUserData callbackData);

  virtual void populateDelayedSubmission();

  /**
   * @brief Callback executed by UCX when a tag send request is completed.
   *
   * Callback executed by UCX when a tag send request is completed, that will dispatch
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
  static void memPutCallback(void* request, ucs_status_t status, void* arg);

  /**
   * @brief Callback executed by UCX when a tag receive request is completed.
   *
   * Callback executed by UCX when a tag receive request is completed, that will dispatch
   * `ucxx::RequestTag::callback()`.
   *
   * WARNING: This is not intended to be called by the user, but it currently needs to be
   * a public method so that UCX may access it. In future changes this will be moved to
   * an internal object and remove this method from the public API.
   *
   * @param[in] request the UCX request pointer.
   * @param[in] status  the completion status of the request.
   * @param[in] info    information of the completed transfer provided by UCX, includes
   *                    length of message received used to verify for truncation.
   * @param[in] arg     the pointer to the `ucxx::Request` object that created the
   *                    transfer, effectively `this` pointer as seen by `request()`.
   */
  static void memGetCallback(void* request, ucs_status_t status, void* arg);

  /**
   * @brief Create and submit an active message send request.
   *
   * This is the method that should be called to actually submit an active message send
   * request. It is meant to be called from `populateDelayedSubmission()`, which is decided
   * at the discretion of `std::shared_ptr<ucxx::Worker>`. See `populateDelayedSubmission()`
   * for more details.
   */
  void request();

  //   /**
  //    * @brief Receive callback registered by `ucxx::Worker`.
  //    *
  //    * This is the receive callback registered by the `ucxx::Worker` to handle incoming active
  //    * messages. For each incoming active message, a proper buffer will be allocated based on
  //    * the header sent by the remote endpoint using the default allocator or one registered by
  //    * the user via `ucxx::Worker::registerAmAllocator()`. Following that, the message is
  //    * immediately received onto the buffer and a `UCS_OK` or the proper error status is set
  //    * onto the `RequestAm` that is returned to the user, or will be later handled by another
  //    * callback when the message is ready. If the callback is executed when a user has already
  //    * requested received of the active message, the buffer and status will be set on the
  //    * earliest request, otherwise a new request is created and saved in a pool that will be
  //    * already populated and ready for consumption or waiting for the internal callback when
  //    * requested.
  //    *
  //    * This is always called by `ucp_worker_progress()`, and thus will happen in the same
  //    * thread that is called from, when using the worker progress thread, this is called from
  //    * the progress thread.
  //    *
  //    * param[in,out] arg  pointer to the `AmData` object held by the `ucxx::Worker` who
  //    *                    registered this callback.
  //    * param[in] header pointer to the header containing the sender buffer's memory type.
  //    * param[in] header_length  length in bytes of the receive header.
  //    * param[in] data pointer to the buffer containing the remote endpoint's send data.
  //    * param[in] length the length in bytes of the message to be received.
  //    * param[in] param  UCP parameters of the active message being received.
  //    */
  //   static ucs_status_t recvCallback(void* arg,
  //                                    const void* header,
  //                                    size_t header_length,
  //                                    void* data,
  //                                    size_t length,
  //                                    const ucp_am_recv_param_t* param);
};

}  // namespace ucxx
