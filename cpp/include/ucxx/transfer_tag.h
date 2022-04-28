/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

#include <ucxx/transfer_common.h>
#include <ucxx/typedefs.h>

#ifdef UCXX_ENABLE_PYTHON
#include <ucxx/python/future.h>
#endif

namespace ucxx {

static void tag_send_callback(void* request, ucs_status_t status, void* arg)
{
  ucxx_trace_req("tag_send_callback");
  return _callback(request, status, arg, std::string{"tag_send"});
}

static void tag_recv_callback(void* request,
                              ucs_status_t status,
                              const ucp_tag_recv_info_t* info,
                              void* arg)
{
  ucxx_trace_req("tag_recv_callback");
  return _callback(request, status, arg, std::string{"tag_recv"});
}

ucs_status_ptr_t tag_request(ucp_worker_h worker,
                             ucp_ep_h ep,
                             bool send,
                             void* buffer,
                             size_t length,
                             ucp_tag_t tag,
                             ucxx_request_t* request)
{
  static const ucp_tag_t tag_mask = -1;

  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_DATATYPE |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = request};

  if (send) {
    param.cb.send = tag_send_callback;
    return ucp_tag_send_nbx(ep, buffer, length, tag, &param);
  } else {
    param.cb.recv = tag_recv_callback;
    return ucp_tag_recv_nbx(worker, buffer, length, tag, tag_mask, &param);
  }
}

typedef struct {
  ucp_worker_h worker     = nullptr;
  ucp_ep_h ep             = nullptr;
  bool send               = false;
  void* buffer            = nullptr;
  size_t length           = 0;
  ucp_tag_t tag           = 0;
  ucxx_request_t* request = nullptr;

} delayed_notification_request_t;

void populate_delayed_notification_request(std::shared_ptr<void> delayed_notification_request)
{
  auto data =
    std::static_pointer_cast<delayed_notification_request_t>(delayed_notification_request);

  std::string operationName{data->send ? "tag_send" : "tag_recv"};
  void* status = tag_request(
    data->worker, data->ep, data->send, data->buffer, data->length, data->tag, data->request);
  ucxx_trace_req("%s request: %p, tag: %lx, buffer: %p, size: %lu",
                 operationName.c_str(),
                 status,
                 data->tag,
                 data->buffer,
                 data->length);
  request_wait(data->worker, status, data->request, operationName);
}

std::shared_ptr<ucxx_request_t> tag_msg(std::shared_ptr<UCXXWorker> worker,
                                        ucp_ep_h ep,
                                        bool send,
                                        void* buffer,
                                        size_t length,
                                        ucp_tag_t tag,
                                        void* callbackFunction             = nullptr,
                                        std::shared_ptr<void> callbackData = nullptr)
{
  auto request = std::make_shared<ucxx_request_t>();
#ifdef UCXX_ENABLE_PYTHON
  request->py_future = create_python_future();
#endif
  request->callback      = callbackFunction;
  request->callback_data = callbackData;

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  auto delayed_notification_request    = std::make_shared<delayed_notification_request_t>();
  delayed_notification_request->worker = worker->get_handle();
  delayed_notification_request->ep     = ep;
  delayed_notification_request->send   = send;
  delayed_notification_request->buffer = buffer;
  delayed_notification_request->length = length;
  delayed_notification_request->tag    = tag;
  // TODO: Fix passing shared_ptr instead of raw pointer, this may be dangerous in this context
  delayed_notification_request->request = request.get();
  worker->registerDelayedNotificationRequest(populate_delayed_notification_request,
                                             delayed_notification_request);

  return request;
}

}  // namespace ucxx
