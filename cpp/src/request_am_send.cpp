/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request_am_send.h>
#include <ucxx/request_data.h>

namespace ucxx {

static void amSendCallback(void* request, ucs_status_t status, void* user_data)
{
  Request* req = reinterpret_cast<Request*>(user_data);
  ucxx_trace_req_f(req->getOwnerString().c_str(), nullptr, request, "amSend", "amSendCallback");
  req->callback(request, status);
}

std::shared_ptr<RequestAmSend> createRequestAmSend(
  std::shared_ptr<Endpoint> endpoint,
  const data::AmSend& requestData,
  const bool enablePythonFuture                = false,
  RequestCallbackUserFunction callbackFunction = nullptr,
  RequestCallbackUserData callbackData         = nullptr)
{
  auto req = std::shared_ptr<RequestAmSend>(new RequestAmSend(endpoint,
                                                              requestData,
                                                              std::move("amSend"),
                                                              enablePythonFuture,
                                                              callbackFunction,
                                                              callbackData));

  // Delayed submission lets the worker progress thread call ucp_am_send_nbx, avoiding
  // the need to hold the GIL in the calling thread.
  req->_worker->registerDelayedSubmission(
    req, std::bind(std::mem_fn(&Request::populateDelayedSubmission), req.get()));

  return req;
}

RequestAmSend::RequestAmSend(std::shared_ptr<Component> endpoint,
                             const data::AmSend& requestData,
                             std::string operationName,
                             const bool enablePythonFuture,
                             RequestCallbackUserFunction callbackFunction,
                             RequestCallbackUserData callbackData)
  : Request(endpoint,
            data::RequestData{requestData},
            std::move(operationName),
            enablePythonFuture,
            callbackFunction,
            callbackData)
{
  if (_endpoint == nullptr) throw ucxx::Error("An endpoint is required to send active messages");
}

void RequestAmSend::cancel()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  if (_status == UCS_INPROGRESS) {
    setStatus(UCS_ERR_CANCELED);
  } else {
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "already completed with status: %d (%s)",
                     _status,
                     ucs_status_string(_status));
  }
}

void RequestAmSend::request()
{
  const auto& amSend = std::get<data::AmSend>(_requestData);

  ucp_request_param_t param = {
    .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE |
                    UCP_OP_ATTR_FIELD_FLAGS | UCP_OP_ATTR_FIELD_USER_DATA,
    .flags     = amSend._flags,
    .datatype  = amSend._datatype,
    .user_data = this,
  };
  param.cb.send = amSendCallback;

  const void* sendBuffer = (amSend._datatype == UCP_DATATYPE_IOV)
                             ? reinterpret_cast<const void*>(amSend._iov.data())
                             : amSend._buffer;

  void* request = ucp_am_send_nbx(_endpoint->getHandle(),
                                  amSend._id,
                                  _headerBytes.data(),
                                  _headerBytes.size(),
                                  sendBuffer,
                                  amSend._count,
                                  &param);

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  _request = request;
}

void RequestAmSend::populateDelayedSubmission()
{
  if (_endpoint->getHandle() == nullptr) {
    ucxx_warn("Endpoint was closed before AM send could be submitted");
    Request::callback(this, UCS_ERR_CANCELED);
    return;
  }

  const auto& amSend = std::get<data::AmSend>(_requestData);

  // Copy header bytes so caller's pointer doesn't need to remain valid.
  if (amSend._header != nullptr && amSend._headerLength > 0) {
    _headerBytes.resize(amSend._headerLength);
    std::memcpy(_headerBytes.data(), amSend._header, amSend._headerLength);
  }

  request();

  if (_enablePythonFuture)
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "populateDelayedSubmission, buffer %p, size %lu, id: %u, "
                     "future %p, future handle %p",
                     amSend._buffer,
                     amSend._count,
                     amSend._id,
                     _future.get(),
                     _future->getHandle());
  else
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "populateDelayedSubmission, buffer %p, size %lu, id: %u",
                     amSend._buffer,
                     amSend._count,
                     amSend._id);

  process();
}

}  // namespace ucxx
