/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

#include <ucxx/notification_request.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>

#if UCXX_ENABLE_PYTHON
#include <ucxx/python/future.h>
#endif

namespace ucxx {

class UCXXRequestTag : public UCXXRequest {
 private:
  UCXXRequestTag(std::shared_ptr<UCXXWorker> worker,
                 std::shared_ptr<UCXXEndpoint> endpoint,
                 bool send,
                 void* buffer,
                 size_t length,
                 ucp_tag_t tag,
                 std::function<void(std::shared_ptr<void>)> callbackFunction = nullptr,
                 std::shared_ptr<void> callbackData                          = nullptr)
    : UCXXRequest(endpoint,
                  UCXXRequest::createRequestBase(worker, callbackFunction, callbackData),
                  std::make_shared<NotificationRequest>(
                    worker->get_handle(), endpoint->getHandle(), send, buffer, length, tag))
  {
    // A delayed notification request is not populated immediately, instead it is
    // delayed to allow the worker progress thread to set its status, and more
    // importantly the Python future later on, so that we don't need the GIL here.
    worker->registerNotificationRequest(
      std::bind(std::mem_fn(&UCXXRequest::populateNotificationRequest), this));
  }

 public:
  static void tag_send_callback(void* request, ucs_status_t status, void* arg)
  {
    ucxx_trace_req("tag_send_callback");
    return UCXXRequest::callback(request, status, arg, std::string{"tag_send"});
  }

  static void tag_recv_callback(void* request,
                                ucs_status_t status,
                                const ucp_tag_recv_info_t* info,
                                void* arg)
  {
    ucxx_trace_req("tag_recv_callback");
    return UCXXRequest::callback(request, status, arg, std::string{"tag_recv"});
  }

  static ucs_status_ptr_t request(ucp_worker_h worker,
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

  virtual void populateNotificationRequest()
  {
    auto data = _notificationRequest;

    std::string operationName{data->_send ? "tag_send" : "tag_recv"};
    void* status = UCXXRequestTag::request(data->_worker,
                                           data->_ep,
                                           data->_send,
                                           data->_buffer,
                                           data->_length,
                                           data->_tag,
                                           _handle.get());
#if UCXX_ENABLE_PYTHON
    ucxx_trace_req("%s request: %p, tag: %lx, buffer: %p, size: %lu, future: %p, future handle: %p",
                   operationName.c_str(),
                   status,
                   data->_tag,
                   data->_buffer,
                   data->_length,
                   _handle->py_future.get(),
                   _handle->py_future->getHandle());
#else
    ucxx_trace_req("%s request: %p, tag: %lx, buffer: %p, size: %lu",
                   operationName.c_str(),
                   status,
                   data->_tag,
                   data->_buffer,
                   data->_length);
#endif
    process(data->_worker, status, operationName);
  }

  friend std::shared_ptr<UCXXRequestTag> createRequestTag(
    std::shared_ptr<UCXXWorker> worker,
    std::shared_ptr<UCXXEndpoint> endpoint,
    bool send,
    void* buffer,
    size_t length,
    ucp_tag_t tag,
    std::function<void(std::shared_ptr<void>)> callbackFunction,
    std::shared_ptr<void> callbackData);
};

}  // namespace ucxx
