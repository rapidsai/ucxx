/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucxx/notification_request.h>
#include <ucxx/request_stream.h>

namespace ucxx {

UCXXRequestStream::UCXXRequestStream(std::shared_ptr<UCXXEndpoint> endpoint,
                                     bool send,
                                     void* buffer,
                                     size_t length)
  : UCXXRequest(endpoint,
                std::make_shared<NotificationRequest>(send, buffer, length),
                std::string(send ? "stream_send" : "stream_recv"))
{
  auto worker = UCXXEndpoint::getWorker(endpoint->getParent());

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  worker->registerNotificationRequest(
    std::bind(std::mem_fn(&UCXXRequest::populateNotificationRequest), this));
}

std::shared_ptr<UCXXRequestStream> createRequestStream(std::shared_ptr<UCXXEndpoint> endpoint,
                                                       bool send,
                                                       void* buffer,
                                                       size_t length)
{
  return std::shared_ptr<UCXXRequestStream>(new UCXXRequestStream(endpoint, send, buffer, length));
}

void UCXXRequestStream::request()
{
  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_DATATYPE |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = this};

  if (_notificationRequest->_send) {
    param.cb.send     = stream_send_callback;
    _requestStatusPtr = ucp_stream_send_nbx(
      _endpoint->getHandle(), _notificationRequest->_buffer, _notificationRequest->_length, &param);
  } else {
    param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
    param.flags          = UCP_STREAM_RECV_FLAG_WAITALL;
    param.cb.recv_stream = stream_recv_callback;
    _requestStatusPtr    = ucp_stream_recv_nbx(_endpoint->getHandle(),
                                            _notificationRequest->_buffer,
                                            _notificationRequest->_length,
                                            &_notificationRequest->_length,
                                            &param);
  }
}

void UCXXRequestStream::populateNotificationRequest()
{
  auto data = _notificationRequest;

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

void UCXXRequestStream::stream_send_callback(void* request, ucs_status_t status, void* arg)
{
  ucxx_trace_req("stream_send_callback");
  UCXXRequest* req = (UCXXRequest*)arg;
  return req->callback(request, status);
}

void UCXXRequestStream::stream_recv_callback(void* request,
                                             ucs_status_t status,
                                             size_t length,
                                             void* arg)
{
  ucxx_trace_req("stream_recv_callback");
  UCXXRequest* req = (UCXXRequest*)arg;
  return req->callback(request, status);
}

}  // namespace ucxx
