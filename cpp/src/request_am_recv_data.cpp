/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <ucp/api/ucp.h>

#include <ucxx/request_am_recv_data.h>
#include <ucxx/request_data.h>

namespace ucxx {

static void amRecvDataCallback(void* request,
                               ucs_status_t status,
                               size_t /* length */,
                               void* user_data)
{
  Request* req = reinterpret_cast<Request*>(user_data);
  ucxx_trace_req_f(
    req->getOwnerString().c_str(), nullptr, request, "amRecvData", "amRecvDataCallback");
  req->callback(request, status);
}

std::shared_ptr<RequestAmRecvData> createRequestAmRecvData(
  std::shared_ptr<Worker> worker,
  const data::AmRecvData& requestData,
  const bool enablePythonFuture                = false,
  RequestCallbackUserFunction callbackFunction = nullptr,
  RequestCallbackUserData callbackData         = nullptr)
{
  auto req = std::shared_ptr<RequestAmRecvData>(new RequestAmRecvData(worker,
                                                                      requestData,
                                                                      std::move("amRecvData"),
                                                                      enablePythonFuture,
                                                                      callbackFunction,
                                                                      callbackData));

  // ucp_am_recv_data_nbx must be called before the AM handler callback returns, so we
  // submit synchronously here rather than going through the delayed submission queue.
  req->request();
  req->process();

  static_cast<void>(worker->registerInflightRequest(req));
  return req;
}

RequestAmRecvData::RequestAmRecvData(std::shared_ptr<Component> worker,
                                     const data::AmRecvData& requestData,
                                     std::string operationName,
                                     const bool enablePythonFuture,
                                     RequestCallbackUserFunction callbackFunction,
                                     RequestCallbackUserData callbackData)
  : Request(worker,
            data::RequestData{requestData},
            std::move(operationName),
            enablePythonFuture,
            callbackFunction,
            callbackData)
{
  if (_worker == nullptr) throw ucxx::Error("A worker is required to receive AM data");
}

void RequestAmRecvData::cancel()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  if (_status == UCS_INPROGRESS) {
    if (_request != nullptr && UCS_PTR_IS_PTR(_request)) {
      ucp_request_cancel(_worker->getHandle(), _request);
    } else {
      setStatus(UCS_ERR_CANCELED);
    }
  }
}

void RequestAmRecvData::request()
{
  const auto& recv = std::get<data::AmRecvData>(_requestData);

  ucp_request_param_t param = {
    .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE |
                    UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL,
    .datatype  = recv._datatype,
    .user_data = this,
  };
  param.cb.recv_am = amRecvDataCallback;

  void* request =
    ucp_am_recv_data_nbx(_worker->getHandle(), recv._dataDesc, recv._buffer, recv._count, &param);

  ucxx_trace_req_f(_ownerString.c_str(),
                   this,
                   request,
                   _operationName.c_str(),
                   "submitted, dataDesc: %p, buffer: %p, count: %lu",
                   recv._dataDesc,
                   recv._buffer,
                   recv._count);

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  _request = request;
}

void RequestAmRecvData::populateDelayedSubmission()
{
  // Should never be called: RequestAmRecvData is always submitted synchronously.
  throw std::logic_error(
    "RequestAmRecvData::populateDelayedSubmission must not be called, "
    "this request is submitted synchronously in createRequestAmRecvData");
}

}  // namespace ucxx
