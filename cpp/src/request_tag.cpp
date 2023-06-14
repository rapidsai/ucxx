/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request_tag.h>

namespace ucxx {

std::shared_ptr<RequestTag> createRequestTag(std::shared_ptr<Component> endpointOrWorker,
                                             bool send,
                                             void* buffer,
                                             size_t length,
                                             ucp_tag_t tag,
                                             const bool enablePythonFuture                = false,
                                             RequestCallbackUserFunction callbackFunction = nullptr,
                                             RequestCallbackUserData callbackData         = nullptr)
{
  auto req = std::shared_ptr<RequestTag>(new RequestTag(endpointOrWorker,
                                                        send,
                                                        buffer,
                                                        length,
                                                        tag,
                                                        enablePythonFuture,
                                                        callbackFunction,
                                                        callbackData));

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  req->_worker->registerDelayedSubmission(
    req, std::bind(std::mem_fn(&Request::populateDelayedSubmission), req.get()));

  return req;
}

RequestTag::RequestTag(std::shared_ptr<Component> endpointOrWorker,
                       bool send,
                       void* buffer,
                       size_t length,
                       ucp_tag_t tag,
                       const bool enablePythonFuture,
                       RequestCallbackUserFunction callbackFunction,
                       RequestCallbackUserData callbackData)
  : Request(endpointOrWorker,
            std::make_shared<DelayedSubmission>(send, buffer, length, tag),
            std::string(send ? "tagSend" : "tagRecv"),
            enablePythonFuture),
    _length(length)
{
  if (send && _endpoint == nullptr)
    throw ucxx::Error("An endpoint is required to send tag messages");
  _callback     = callbackFunction;
  _callbackData = callbackData;
}

void RequestTag::callback(void* request, ucs_status_t status, const ucp_tag_recv_info_t* info)
{
  if (status != UCS_ERR_CANCELED && info->length != _length) {
    status          = UCS_ERR_MESSAGE_TRUNCATED;
    const char* fmt = "length mismatch: %llu (got) != %llu (expected)";
    size_t len      = std::snprintf(nullptr, 0, fmt, info->length, _length);
    _status_msg     = std::string(len + 1, '\0');  // +1 for null terminator
    std::snprintf(_status_msg.data(), _status_msg.size(), fmt, info->length, _length);
  }

  Request::callback(request, status);
}

void RequestTag::tagSendCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), request, "tagSend", "tagSendCallback");
  return req->callback(request, status);
}

void RequestTag::tagRecvCallback(void* request,
                                 ucs_status_t status,
                                 const ucp_tag_recv_info_t* info,
                                 void* arg)
{
  RequestTag* req = reinterpret_cast<RequestTag*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), request, "tagRecv", "tagRecvCallback");
  return req->callback(request, status, info);
}

void RequestTag::request()
{
  static const ucp_tag_t tagMask = -1;

  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_DATATYPE |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = this};

  if (_delayedSubmission->_send) {
    param.cb.send = tagSendCallback;
    _request      = ucp_tag_send_nbx(_endpoint->getHandle(),
                                _delayedSubmission->_buffer,
                                _delayedSubmission->_length,
                                _delayedSubmission->_tag,
                                &param);
  } else {
    param.cb.recv = tagRecvCallback;
    _request      = ucp_tag_recv_nbx(_worker->getHandle(),
                                _delayedSubmission->_buffer,
                                _delayedSubmission->_length,
                                _delayedSubmission->_tag,
                                tagMask,
                                &param);
  }
}

void RequestTag::populateDelayedSubmission()
{
  request();

  if (_enablePythonFuture)
    ucxx_trace_req_f(
      _ownerString.c_str(),
      _request,
      _operationName.c_str(),
      "tag 0x%lx, buffer %p, size %lu, future %p, future handle %p, populateDelayedSubmission",
      _delayedSubmission->_tag,
      _delayedSubmission->_buffer,
      _delayedSubmission->_length,
      _future.get(),
      _future->getHandle());
  else
    ucxx_trace_req_f(_ownerString.c_str(),
                     _request,
                     _operationName.c_str(),
                     "tag 0x%lx, buffer %p, size %lu, populateDelayedSubmission",
                     _delayedSubmission->_tag,
                     _delayedSubmission->_buffer,
                     _delayedSubmission->_length);

  process();
}

}  // namespace ucxx
