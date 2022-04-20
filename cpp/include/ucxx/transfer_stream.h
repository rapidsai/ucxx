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

namespace ucxx
{

static void stream_send_callback(void *request, ucs_status_t status, void *arg)
{
    ucxx_trace_req("stream_send_callback");
    return _callback(request, status, arg, std::string{"stream_send"});
}

static void stream_recv_callback(void *request, ucs_status_t status, size_t length, void *arg)
{
    ucxx_trace_req("stream_recv_callback");
    return _callback(request, status, arg, std::string{"stream_recv"});
}

ucs_status_ptr_t stream_request(ucp_ep_h ep,
                                bool send, void *buffer, size_t length,
                                ucxx_request_t* request)
{
    ucp_request_param_t param = {
        .op_attr_mask               = UCP_OP_ATTR_FIELD_CALLBACK |
                                      UCP_OP_ATTR_FIELD_DATATYPE |
                                      UCP_OP_ATTR_FIELD_USER_DATA,
        .datatype                   = ucp_dt_make_contig(1),
        .user_data                  = request
    };

    if (send)
    {
        param.cb.send = stream_send_callback;
        return ucp_stream_send_nbx(ep, buffer, length, &param);
    }
    else
    {
        param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
        param.flags = UCP_STREAM_RECV_FLAG_WAITALL;
        param.cb.recv_stream = stream_recv_callback;
        return ucp_stream_recv_nbx(ep, buffer, length, &length, &param);
    }
}

std::shared_ptr<ucxx_request_t> stream_msg(ucp_worker_h worker, ucp_ep_h ep,
             bool send, void* buffer, size_t length)
{
    std::shared_ptr<ucxx_request_t> request = std::make_shared<ucxx_request_t>();
#ifdef UCXX_ENABLE_PYTHON
    request->py_future = create_python_future();
#endif
    std::string operationName{send ? "stream_send" : "stream_recv"};
    void* status = stream_request(ep, send, buffer, length, request.get());
    ucxx_trace_req("%s request: %p, buffer: %p, size: %lu",
                   operationName.c_str(), status, buffer, length);
    request_wait(worker, status, request.get(), operationName);
    return request;
}

}  // namespace ucxx
