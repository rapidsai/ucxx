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

class UCXXRequestStream : public UCXXRequest {
 private:
  UCXXRequestStream(std::shared_ptr<UCXXEndpoint> endpoint,
                    inflight_requests_t inflightRequests,
                    std::shared_ptr<ucxx_request_t> request)
    : UCXXRequest(endpoint, inflightRequests, request)
  {
  }

 public:
  static void stream_send_callback(void* request, ucs_status_t status, void* arg)
  {
    ucxx_trace_req("stream_send_callback");
    return UCXXRequest::callback(request, status, arg, std::string{"stream_send"});
  }

  static void stream_recv_callback(void* request, ucs_status_t status, size_t length, void* arg)
  {
    ucxx_trace_req("stream_recv_callback");
    return UCXXRequest::callback(request, status, arg, std::string{"stream_recv"});
  }

  static ucs_status_ptr_t stream_request(
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

  static void populateNotificationRequestStream(
    std::shared_ptr<NotificationRequest> notificationRequest)
  {
    auto data = notificationRequest;

    std::string operationName{data->_send ? "stream_send" : "stream_recv"};
    void* status =
      stream_request(data->_ep, data->_send, data->_buffer, data->_length, data->_request.get());
#if UCXX_ENABLE_PYTHON
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
    UCXXRequest::process(data->_worker, status, data->_request.get(), operationName);
  }

  friend std::shared_ptr<UCXXRequestStream> createRequestStream(
    std::shared_ptr<UCXXWorker> worker,
    std::shared_ptr<UCXXEndpoint> endpoint,
    bool send,
    void* buffer,
    size_t length);
};

}  // namespace ucxx
