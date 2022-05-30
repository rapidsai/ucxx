/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucxx/notification_request.h>
#include <ucxx/request_tag.h>

#if UCXX_ENABLE_PYTHON
#include <ucxx/python/python_future.h>
#endif

namespace ucxx {

std::shared_ptr<RequestTag> createRequestTag(
  std::shared_ptr<Endpoint> endpoint,
  bool send,
  void* buffer,
  size_t length,
  ucp_tag_t tag,
  const bool enablePythonFuture                               = true,
  std::function<void(std::shared_ptr<void>)> callbackFunction = nullptr,
  std::shared_ptr<void> callbackData                          = nullptr)
{
  return std::shared_ptr<RequestTag>(new RequestTag(
    endpoint, send, buffer, length, tag, enablePythonFuture, callbackFunction, callbackData));
}

RequestTag::RequestTag(std::shared_ptr<Endpoint> endpoint,
                       bool send,
                       void* buffer,
                       size_t length,
                       ucp_tag_t tag,
                       const bool enablePythonFuture,
                       std::function<void(std::shared_ptr<void>)> callbackFunction,
                       std::shared_ptr<void> callbackData)
  : Request(endpoint,
            std::make_shared<NotificationRequest>(send, buffer, length, tag),
            std::string(send ? "tag_send" : "tag_recv"),
            enablePythonFuture)
{
  auto worker = Endpoint::getWorker(endpoint->getParent());

  _handle->callback      = callbackFunction;
  _handle->callback_data = callbackData;

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  worker->registerNotificationRequest(
    std::bind(std::mem_fn(&Request::populateNotificationRequest), this));
}

void RequestTag::tag_send_callback(void* request, ucs_status_t status, void* arg)
{
  ucxx_trace_req("tag_send_callback");
  Request* req = (Request*)arg;
  return req->callback(request, status);
}

void RequestTag::tag_recv_callback(void* request,
                                   ucs_status_t status,
                                   const ucp_tag_recv_info_t* info,
                                   void* arg)
{
  ucxx_trace_req("tag_recv_callback");
  Request* req = (Request*)arg;
  return req->callback(request, status);
}

void RequestTag::request()
{
  static const ucp_tag_t tag_mask = -1;
  auto worker                     = Endpoint::getWorker(_endpoint->getParent());

  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_DATATYPE |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = this};

  if (_notificationRequest->_send) {
    param.cb.send     = tag_send_callback;
    _requestStatusPtr = ucp_tag_send_nbx(_endpoint->getHandle(),
                                         _notificationRequest->_buffer,
                                         _notificationRequest->_length,
                                         _notificationRequest->_tag,
                                         &param);
  } else {
    param.cb.recv     = tag_recv_callback;
    _requestStatusPtr = ucp_tag_recv_nbx(worker->get_handle(),
                                         _notificationRequest->_buffer,
                                         _notificationRequest->_length,
                                         _notificationRequest->_tag,
                                         tag_mask,
                                         &param);
  }
}

void RequestTag::populateNotificationRequest()
{
  const bool pythonFutureLog = _enablePythonFuture & UCXX_ENABLE_PYTHON;

  request();

#if UCXX_ENABLE_PYTHON
  if (pythonFutureLog)
    ucxx_trace_req("%s request: %p, tag: %lx, buffer: %p, size: %lu, future: %p, future handle: %p",
                   _operationName.c_str(),
                   _requestStatusPtr,
                   _notificationRequest->_tag,
                   _notificationRequest->_buffer,
                   _notificationRequest->_length,
                   _handle->py_future.get(),
                   _handle->py_future->getHandle());
#endif
  if (!pythonFutureLog)
    ucxx_trace_req("%s request: %p, tag: %lx, buffer: %p, size: %lu",
                   _operationName.c_str(),
                   _requestStatusPtr,
                   _notificationRequest->_tag,
                   _notificationRequest->_buffer,
                   _notificationRequest->_length);
  process();
}

}  // namespace ucxx
