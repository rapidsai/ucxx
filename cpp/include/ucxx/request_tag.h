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
  UCXXRequestTag(std::shared_ptr<UCXXEndpoint> endpoint,
                 bool send,
                 void* buffer,
                 size_t length,
                 ucp_tag_t tag,
                 std::function<void(std::shared_ptr<void>)> callbackFunction = nullptr,
                 std::shared_ptr<void> callbackData                          = nullptr)
    : UCXXRequest(endpoint,
                  std::make_shared<NotificationRequest>(send, buffer, length, tag),
                  std::string(send ? "tag_send" : "tag_recv"))
  {
    auto worker = UCXXEndpoint::getWorker(endpoint->getParent());

    _handle->callback      = callbackFunction;
    _handle->callback_data = callbackData;

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
    UCXXRequest* req = (UCXXRequest*)arg;
    return req->callback(request, status);
  }

  static void tag_recv_callback(void* request,
                                ucs_status_t status,
                                const ucp_tag_recv_info_t* info,
                                void* arg)
  {
    ucxx_trace_req("tag_recv_callback");
    UCXXRequest* req = (UCXXRequest*)arg;
    return req->callback(request, status);
  }

  ucs_status_ptr_t request()
  {
    static const ucp_tag_t tag_mask = -1;
    auto worker                     = UCXXEndpoint::getWorker(_endpoint->getParent());

    ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                                 UCP_OP_ATTR_FIELD_DATATYPE |
                                                 UCP_OP_ATTR_FIELD_USER_DATA,
                                 .datatype  = ucp_dt_make_contig(1),
                                 .user_data = this};

    if (_notificationRequest->_send) {
      param.cb.send = tag_send_callback;
      return ucp_tag_send_nbx(_endpoint->getHandle(),
                              _notificationRequest->_buffer,
                              _notificationRequest->_length,
                              _notificationRequest->_tag,
                              &param);
    } else {
      param.cb.recv = tag_recv_callback;
      return ucp_tag_recv_nbx(worker->get_handle(),
                              _notificationRequest->_buffer,
                              _notificationRequest->_length,
                              _notificationRequest->_tag,
                              tag_mask,
                              &param);
    }
  }

  virtual void populateNotificationRequest()
  {
    auto data = _notificationRequest;

    void* status = request();
#if UCXX_ENABLE_PYTHON
    ucxx_trace_req("%s request: %p, tag: %lx, buffer: %p, size: %lu, future: %p, future handle: %p",
                   _operationName.c_str(),
                   status,
                   _notificationRequest->_tag,
                   _notificationRequest->_buffer,
                   _notificationRequest->_length,
                   _handle->py_future.get(),
                   _handle->py_future->getHandle());
#else
    ucxx_trace_req("%s request: %p, tag: %lx, buffer: %p, size: %lu",
                   _operationName.c_str(),
                   status,
                   _notificationRequest->_tag,
                   _notificationRequest->_buffer,
                   _notificationRequest->_length);
#endif
    process(status);
  }

  friend std::shared_ptr<UCXXRequestTag> createRequestTag(
    std::shared_ptr<UCXXEndpoint> endpoint,
    bool send,
    void* buffer,
    size_t length,
    ucp_tag_t tag,
    std::function<void(std::shared_ptr<void>)> callbackFunction,
    std::shared_ptr<void> callbackData);
};

}  // namespace ucxx
