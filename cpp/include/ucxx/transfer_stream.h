/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

#include <ucxx/delayed_notification_request.h>
#include <ucxx/transfer_common.h>
#include <ucxx/typedefs.h>

#ifdef UCXX_ENABLE_PYTHON
#include <ucxx/python/future.h>
#endif

namespace ucxx {

static void stream_send_callback(void* request, ucs_status_t status, void* arg)
{
  ucxx_trace_req("stream_send_callback");
  return _callback(request, status, arg, std::string{"stream_send"});
}

static void stream_recv_callback(void* request, ucs_status_t status, size_t length, void* arg)
{
  ucxx_trace_req("stream_recv_callback");
  return _callback(request, status, arg, std::string{"stream_recv"});
}

ucs_status_ptr_t stream_request(
  ucp_ep_h ep, bool send, void* buffer, size_t length, ucxx_request_t* request)
{
  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_DATATYPE |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = request};

  if (send) {
    param.cb.send = stream_send_callback;
    return ucp_stream_send_nbx(ep, buffer, length, &param);
  } else {
    param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
    param.flags          = UCP_STREAM_RECV_FLAG_WAITALL;
    param.cb.recv_stream = stream_recv_callback;
    return ucp_stream_recv_nbx(ep, buffer, length, &length, &param);
  }
}

void populate_delayed_notification_stream_request(
  std::shared_ptr<DelayedNotificationRequest> delayedNotificationRequest)
{
  auto data = delayedNotificationRequest;

  std::string operationName{data->_send ? "stream_send" : "stream_recv"};
  void* status =
    stream_request(data->_ep, data->_send, data->_buffer, data->_length, data->_request.get());
#ifdef UCXX_ENABLE_PYTHON
  ucxx_trace_req("%s request: %p, buffer: %p, size: %lu, future: %p, future handle: %p",
                 operationName.c_str(),
                 status,
                 data->_buffer,
                 data->_length,
                 data->_request->py_future.get(),
                 data->_request->py_future->getHandle());
#else
  ucxx_trace_req("%s request: %p, buffer: %p, size: %lu",
                 operationName.c_str(),
                 status,
                 data->_buffer,
                 data->_length);
#endif
  request_wait(data->_worker, status, data->_request.get(), operationName);
}

std::shared_ptr<ucxx_request_t> stream_msg(
  std::shared_ptr<UCXXWorker> worker, ucp_ep_h ep, bool send, void* buffer, size_t length)
{
  auto request = std::make_shared<ucxx_request_t>();
#ifdef UCXX_ENABLE_PYTHON
  request->py_future = worker->getPythonFuture();
  ucxx_trace_req("request: %p, request->py_future: %p", request.get(), request->py_future.get());
#endif
  request->callback      = nullptr;
  request->callback_data = nullptr;

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  auto delayedNotificationRequest = std::make_shared<DelayedNotificationRequest>(
    worker->get_handle(), ep, request, send, buffer, length);
  worker->registerDelayedNotificationRequest(populate_delayed_notification_stream_request,
                                             delayedNotificationRequest);

  return request;
}

}  // namespace ucxx
