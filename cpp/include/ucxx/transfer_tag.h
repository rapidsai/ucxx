/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucxx/typedefs.h>

namespace ucxx
{

static void _callback(void *request, ucs_status_t status, void *arg, std::string operation)
{
    ucxx_request_t* ucxx_req = (ucxx_request_t*)arg;

    if (ucxx_req == nullptr)
        std::cerr << "[0x" << std::hex << std::this_thread::get_id() << "] " <<
            "error when _callback was called for \"" << operation << "\", " <<
            "probably due to tag_msg() return value being deleted before completion." <<
            std::endl;

    std::cout << "[0x" << std::hex << std::this_thread::get_id() << "] _callback called for \"" <<
        operation << "\" with status " << status << " (" << ucs_status_string(status) << ")" <<
        std::endl;

    status = ucp_request_check_status(request);
    ucxx_req->completed_promise.set_value(UCS_OK);

    ucp_request_free(request);
}

static void tag_send_callback(void *request, ucs_status_t status, void *arg)
{
    return _callback(request, status, arg, std::string{"tag_send"});
}

static void tag_recv_callback(void *request, ucs_status_t status, const ucp_tag_recv_info_t *info, void *arg)
{
    return _callback(request, status, arg, std::string{"tag_recv"});
}

static void request_wait(ucp_worker_h worker, void *request,
                                 ucxx_request_t* ucxx_req,
                                 std::string operationName)
{
    ucs_status_t status;

    // Operation completed immediately
    if (request == NULL)
    {
        status = UCS_OK;
    }
    else
    {
        if (UCS_PTR_IS_ERR(request))
            status = UCS_PTR_STATUS(request);
        else if (UCS_PTR_IS_PTR(request))
            // Completion will be handled by callback
            return;
        else
            status = UCS_OK;
    }

    if (status != UCS_OK)
        std::cerr << "error on " << operationName << "(" <<
            ucs_status_string(status) << ")" << std::endl;
    else
        std::cout << operationName << " completed" << std::endl;

    ucxx_req->completed_promise.set_value(status);
    return;
}


ucs_status_ptr_t tag_request(ucp_worker_h worker, ucp_ep_h ep,
                             bool send, void *buffer, size_t length,
                             ucp_tag_t tag, ucxx_request_t* request)
{
    static const ucp_tag_t tag_mask = -1;

    ucp_request_param_t param = {
        .op_attr_mask               = UCP_OP_ATTR_FIELD_CALLBACK |
                                      UCP_OP_ATTR_FIELD_DATATYPE |
                                      UCP_OP_ATTR_FIELD_USER_DATA,
        .datatype                   = ucp_dt_make_contig(1),
        .user_data                  = request
    };

    if (send)
    {
        param.cb.send = tag_send_callback;
        return ucp_tag_send_nbx(ep, buffer, length, tag, &param);
    }
    else
    {
        param.cb.recv = tag_recv_callback;
        return ucp_tag_recv_nbx(worker, buffer, length, tag, tag_mask, &param);
    }
}


std::shared_ptr<ucxx_request_t> tag_msg(ucp_worker_h worker, ucp_ep_h ep,
             bool send, void* buffer, size_t length,
             ucp_tag_t tag)
{
    std::shared_ptr<ucxx_request_t> request = std::make_shared<ucxx_request_t>();
    std::string operationName{send ? "tag_send" : "tag_recv"};
    void* status = tag_request(worker, ep, send, buffer, length, tag, request.get());
    request_wait(worker, status, request.get(), operationName);
    return request;
}

}  // namespace ucxx
