/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>

namespace ucxx {

/**
 * @brief Send an Active Message.
 *
 * Wraps `ucp_am_send_nbx`. The AM handler ID and raw header bytes are passed through
 * to UCX without metadata serialization. The remote worker must have a handler
 * registered for the given AM ID via `Worker::setAmHandler()`.
 */
class RequestAmSend : public Request {
 private:
  std::vector<std::byte> _headerBytes{};  ///< Owned copy of header bytes. Copied from the
                                          ///< caller's pointer in populateDelayedSubmission()
                                          ///< so that the caller's buffer does not need to
                                          ///< outlive the request (mirrors workaround for
                                          ///< https://github.com/openucx/ucx/issues/10424).

  /**
   * @brief Private constructor of `ucxx::RequestAmSend`.
   *
   * @param[in] endpoint            the parent endpoint.
   * @param[in] requestData         AmSend data (AM ID, header, buffer, count, flags, datatype).
   * @param[in] operationName       human-readable name for logging.
   * @param[in] enablePythonFuture  whether a Python future should be created.
   * @param[in] callbackFunction    user-defined callback to call upon completion.
   * @param[in] callbackData        data to pass to `callbackFunction`.
   */
  RequestAmSend(std::shared_ptr<Component> endpoint,
                const data::AmSend& requestData,
                std::string operationName,
                const bool enablePythonFuture                = false,
                RequestCallbackUserFunction callbackFunction = nullptr,
                RequestCallbackUserData callbackData         = nullptr);

 public:
  /**
   * @brief Factory for `std::shared_ptr<ucxx::RequestAmSend>`.
   *
   * @param[in] endpoint            the parent endpoint.
   * @param[in] requestData         AmSend data.
   * @param[in] enablePythonFuture  whether a Python future should be created.
   * @param[in] callbackFunction    user-defined callback to call upon completion.
   * @param[in] callbackData        data to pass to `callbackFunction`.
   *
   * @returns The `shared_ptr<ucxx::RequestAmSend>` object.
   */
  friend std::shared_ptr<RequestAmSend> createRequestAmSend(
    std::shared_ptr<Endpoint> endpoint,
    const data::AmSend& requestData,
    const bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);

  void cancel() override;
  void populateDelayedSubmission() override;

  /**
   * @brief Submit the AM send via `ucp_am_send_nbx`.
   *
   * Called from `populateDelayedSubmission()`. Uses the owned `_headerBytes` copy so the
   * caller's header pointer does not need to remain valid after the initial call.
   */
  void request();
};

}  // namespace ucxx
