/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request_stream.h>

namespace ucxx {

RequestStream::RequestStream(std::shared_ptr<Endpoint> endpoint,
                             bool send,
                             void* buffer,
                             size_t length,
                             const bool enablePythonFuture)
  : Request(endpoint,
            std::make_shared<DelayedSubmission>(send, buffer, length),
            std::string(send ? "streamSend" : "streamRecv"),
            enablePythonFuture),
    _length(length)
{
  auto worker = endpoint->getWorker();

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  worker->registerDelayedSubmission(
    std::bind(std::mem_fn(&Request::populateDelayedSubmission), this));
}

std::shared_ptr<RequestStream> createRequestStream(std::shared_ptr<Endpoint> endpoint,
                                                   bool send,
                                                   void* buffer,
                                                   size_t length,
                                                   const bool enablePythonFuture = false)
{
  auto req = std::shared_ptr<RequestStream>(
    new RequestStream(endpoint, send, buffer, length, enablePythonFuture));

  req->_delayedSubmission->setParent(req);
  return req;
}

void RequestStream::request()
{
  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_DATATYPE |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = this};

  if (_delayedSubmission->_send) {
    param.cb.send = streamSendCallback;
    _request      = ucp_stream_send_nbx(
      _endpoint->getHandle(), _delayedSubmission->_buffer, _delayedSubmission->_length, &param);
  } else {
    param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
    param.flags          = UCP_STREAM_RECV_FLAG_WAITALL;
    param.cb.recv_stream = streamRecvCallback;
    _request             = ucp_stream_recv_nbx(_endpoint->getHandle(),
                                   _delayedSubmission->_buffer,
                                   _delayedSubmission->_length,
                                   &_delayedSubmission->_length,
                                   &param);
  }
}

void RequestStream::populateDelayedSubmission()
{
  request();

  if (_enablePythonFuture)
    ucxx_trace_req_f(_ownerString.c_str(),
                     _request,
                     _operationName.c_str(),
                     "buffer %p, size %lu, future %p, future handle %p, populateDelayedSubmission",
                     _delayedSubmission->_buffer,
                     _delayedSubmission->_length,
                     _future.get(),
                     _future->getHandle());
  else
    ucxx_trace_req_f(_ownerString.c_str(),
                     _request,
                     _operationName.c_str(),
                     "buffer %p, size %lu, populateDelayedSubmission",
                     _delayedSubmission->_buffer,
                     _delayedSubmission->_length);
  process();
}

void RequestStream::callback(void* request, ucs_status_t status, size_t length)
{
  status  = length == _length ? status : UCS_ERR_MESSAGE_TRUNCATED;
  _status = status;

  if (status == UCS_ERR_MESSAGE_TRUNCATED) {
    const char* fmt = "length mismatch: %llu (got) != %llu (expected)";
    size_t len      = std::snprintf(nullptr, 0, fmt, length, _length);
    _status_msg     = std::string(len + 1, '\0');  // +1 for null terminator
    std::snprintf(_status_msg.data(), _status_msg.size(), fmt, length, _length);
  }

  Request::callback(request, status);
}

void RequestStream::streamSendCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), request, "streamSend", "streamSendCallback");
  return req->callback(request, status);
}

void RequestStream::streamRecvCallback(void* request, ucs_status_t status, size_t length, void* arg)
{
  RequestStream* req = reinterpret_cast<RequestStream*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), request, "streamRecv", "streamRecvCallback");
  return req->callback(request, status, length);
}

}  // namespace ucxx
