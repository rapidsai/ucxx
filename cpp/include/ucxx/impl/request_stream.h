/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucxx/notification_request.h>
#include <ucxx/request_stream.h>

#if UCXX_ENABLE_PYTHON
#include <ucxx/python/python_future.h>
#endif

namespace ucxx {

RequestStream::RequestStream(std::shared_ptr<Endpoint> endpoint,
                             bool send,
                             void* buffer,
                             size_t length)
  : Request(endpoint,
            std::make_shared<NotificationRequest>(send, buffer, length),
            std::string(send ? "streamSend" : "streamRecv"))
{
  auto worker = Endpoint::getWorker(endpoint->getParent());

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  worker->registerNotificationRequest(
    std::bind(std::mem_fn(&Request::populateNotificationRequest), this));
}

std::shared_ptr<RequestStream> createRequestStream(std::shared_ptr<Endpoint> endpoint,
                                                   bool send,
                                                   void* buffer,
                                                   size_t length)
{
  return std::shared_ptr<RequestStream>(new RequestStream(endpoint, send, buffer, length));
}

void RequestStream::request()
{
  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_DATATYPE |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = this};

  if (_notificationRequest->_send) {
    param.cb.send     = streamSendCallback;
    _requestStatusPtr = ucp_stream_send_nbx(
      _endpoint->getHandle(), _notificationRequest->_buffer, _notificationRequest->_length, &param);
  } else {
    param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
    param.flags          = UCP_STREAM_RECV_FLAG_WAITALL;
    param.cb.recv_stream = streamRecvCallback;
    _requestStatusPtr    = ucp_stream_recv_nbx(_endpoint->getHandle(),
                                            _notificationRequest->_buffer,
                                            _notificationRequest->_length,
                                            &_notificationRequest->_length,
                                            &param);
  }
}

void RequestStream::populateNotificationRequest()
{
  request();

#if UCXX_ENABLE_PYTHON
  ucxx_trace_req("%s request: %p, buffer: %p, size: %lu, future: %p, future handle: %p",
                 _operationName.c_str(),
                 _requestStatusPtr,
                 _notificationRequest->_buffer,
                 _notificationRequest->_length,
                 _handle->py_future.get(),
                 _handle->py_future->getHandle());
#else
  ucxx_trace_req("%s request: %p, buffer: %p, size: %lu",
                 _operationName.c_str(),
                 _requestStatusPtr,
                 _notificationRequest->_buffer,
                 _notificationRequest->_length);
#endif
  process();
}

void RequestStream::streamSendCallback(void* request, ucs_status_t status, void* arg)
{
  ucxx_trace_req("streamSendCallback");
  Request* req = (Request*)arg;
  return req->callback(request, status);
}

void RequestStream::streamRecvCallback(void* request, ucs_status_t status, size_t length, void* arg)
{
  ucxx_trace_req("streamRecvCallback");
  Request* req = (Request*)arg;
  return req->callback(request, status);
}

}  // namespace ucxx
