/**
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
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

/**
 * @brief Send or receive a message with the UCX Tag API.
 *
 * Close a UCP endpoint, using non-blocking UCP call `ucp_ep_close_nbx`.
 */
class RequestEndpointClose : public Request {
 private:
  /**
   * @brief Private constructor of `ucxx::RequestEndpointClose`.
   *
   * This is the internal implementation of `ucxx::RequestEndpointClose` constructor, made
   * private not to be called directly. This constructor is made private to ensure all UCXX
   * objects are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Endpoint::close()`
   * - `ucxx::createRequestEndpointClose()`
   *
   * @throws ucxx::Error  if `endpoint` is not a valid `std::shared_ptr<ucxx::Endpoint>`.
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] requestData         container of the specified message type, including all
   *                                type-specific data.
   * @param[in] operationName       a human-readable operation name to help identifying
   *                                requests by their types when UCXX logging is enabled.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   */
  RequestEndpointClose(std::shared_ptr<Endpoint> endpoint,
                       const data::EndpointClose requestData,
                       const std::string operationName,
                       const bool enablePythonFuture                = false,
                       RequestCallbackUserFunction callbackFunction = nullptr,
                       RequestCallbackUserData callbackData         = nullptr);

 public:
  /**
   * @brief Constructor for `std::shared_ptr<ucxx::RequestEndpointClose>`.
   *
   * The constructor for a `std::shared_ptr<ucxx::RequestEndpointClose>` object, creating a
   * request to close a UCP endpoint, returning a pointer to the request object that can be
   * later awaited and checked for errors. This is a non-blocking operation, and its status
   * must be verified from the resulting request object to confirm the close operation has
   * completed successfully.
   *
   * @throws ucxx::Error  `endpoint` is not a valid `std::shared_ptr<ucxx::Endpoint>`.
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] requestData         container of the specified message type, including all
   *                                type-specific data.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   *
   * @returns The `shared_ptr<ucxx::RequestEndpointClose>` object.
   */
  [[nodiscard]] friend std::shared_ptr<RequestEndpointClose> createRequestEndpointClose(
    std::shared_ptr<Endpoint> endpoint,
    const data::EndpointClose requestData,
    const bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);

  virtual void populateDelayedSubmission();

  /**
   * @brief Create and submit an endpoint close request.
   *
   * This is the method that should be called to actually submit an endpoint close request.
   * It is meant to be called from `populateDelayedSubmission()`, which is decided at the
   * discretion of `std::shared_ptr<ucxx::Worker>`. See `populateDelayedSubmission()` for
   * more details.
   */
  void request();

  /**
   * @brief Callback executed by UCX when an endpoint close request is completed.
   *
   * Callback executed by UCX when an endpoint close request is completed, that will
   * dispatch `ucxx::Request::callback()`.
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
  static void endpointCloseCallback(void* request, ucs_status_t status, void* arg);
};

}  // namespace ucxx
