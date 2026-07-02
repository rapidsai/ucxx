/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/request.h>
#include <ucxx/typedefs.h>

namespace ucxx {

/**
 * @brief Receive rendezvous Active Message data.
 *
 * Wraps `ucp_am_recv_data_nbx`. Must be created and submitted from inside an
 * `AmHandlerType` callback before returning `UCS_INPROGRESS`, because the `dataDesc`
 * pointer passed to the callback is only valid until `ucp_am_recv_data_nbx` is called.
 *
 * Unlike other UCXX requests, `RequestAmRecvData` does not use the `DelayedSubmission`
 * mechanism: `ucp_am_recv_data_nbx` is called synchronously inside `createRequestAmRecvData`.
 */
class RequestAmRecvData : public Request {
 private:
  /**
   * @brief Private constructor of `ucxx::RequestAmRecvData`.
   *
   * @param[in] worker              the parent worker.
   * @param[in] requestData         AmRecvData (dataDesc, buffer, count, datatype).
   * @param[in] operationName       human-readable name for logging.
   * @param[in] enablePythonFuture  whether a Python future should be created.
   * @param[in] callbackFunction    user-defined callback to call upon completion.
   * @param[in] callbackData        data to pass to `callbackFunction`.
   */
  RequestAmRecvData(std::shared_ptr<Component> worker,
                    const data::AmRecvData& requestData,
                    std::string operationName,
                    const bool enablePythonFuture                = false,
                    RequestCallbackUserFunction callbackFunction = nullptr,
                    RequestCallbackUserData callbackData         = nullptr);

 public:
  /**
   * @brief Factory for `std::shared_ptr<ucxx::RequestAmRecvData>`.
   *
   * Calls `ucp_am_recv_data_nbx` synchronously (no delayed submission). Must be called
   * from inside the AM handler callback before returning `UCS_INPROGRESS`.
   *
   * @param[in] worker              the parent worker.
   * @param[in] requestData         AmRecvData (dataDesc, buffer, count, datatype).
   * @param[in] enablePythonFuture  whether a Python future should be created.
   * @param[in] callbackFunction    user-defined callback to call upon completion.
   * @param[in] callbackData        data to pass to `callbackFunction`.
   *
   * @returns The `shared_ptr<ucxx::RequestAmRecvData>` object.
   */
  friend std::shared_ptr<RequestAmRecvData> createRequestAmRecvData(
    std::shared_ptr<Worker> worker,
    const data::AmRecvData& requestData,
    const bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);

  void cancel() override;
  void populateDelayedSubmission() override;

  /**
   * @brief Submit `ucp_am_recv_data_nbx` synchronously.
   *
   * Called directly from `createRequestAmRecvData` (not via the delayed submission queue)
   * because `dataDesc` is only valid until `ucp_am_recv_data_nbx` is called.
   */
  void request();
};

}  // namespace ucxx
