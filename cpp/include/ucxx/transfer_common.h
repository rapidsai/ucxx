/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

#include <ucxx/typedefs.h>

#if UCXX_ENABLE_PYTHON
#include <ucxx/python/python_future.h>
#endif

namespace ucxx {

void set_ucxx_request_status(ucxx_request_t* ucxx_req, ucs_status_t status)
{
  ucxx_req->status = status;

#if UCXX_ENABLE_PYTHON
  auto future = std::static_pointer_cast<PythonFuture>(ucxx_req->py_future);
  future->notify(status);
#endif
}

static void _callback(void* request, ucs_status_t status, void* arg, std::string operation)
{
  ucxx_request_t* ucxx_req = (ucxx_request_t*)arg;
  status                   = ucp_request_check_status(request);

  if (ucxx_req == nullptr)
    ucxx_error(
      "error when _callback was called for \"%s\", "
      "probably due to tag_msg() return value being deleted "
      "before completion.",
      operation.c_str());

  ucxx_trace_req("_calback called for \"%s\" with status %d (%s)",
                 operation.c_str(),
                 status,
                 ucs_status_string(status));

  set_ucxx_request_status(ucxx_req, ucp_request_check_status(request));

  ucxx_trace_req("ucxx_req->callback: %p", ucxx_req->callback.target<void (*)(void)>());
  if (ucxx_req->callback) ucxx_req->callback(ucxx_req->callback_data);

  ucp_request_free(request);
}

static void request_wait(ucp_worker_h worker,
                         void* request,
                         ucxx_request_t* ucxx_req,
                         std::string operationName)
{
  ucs_status_t status;

  // Operation completed immediately
  if (request == NULL) {
    status = UCS_OK;
  } else {
    if (UCS_PTR_IS_ERR(request)) {
      status = UCS_PTR_STATUS(request);
    } else if (UCS_PTR_IS_PTR(request)) {
      // Completion will be handled by callback
      ucxx_req->request = request;
      return;
    } else {
      status = UCS_OK;
    }
  }

  set_ucxx_request_status(ucxx_req, status);

  ucxx_trace_req("ucxx_req->callback: %p", ucxx_req->callback.target<void (*)(void)>());
  if (ucxx_req->callback) ucxx_req->callback(ucxx_req->callback_data);

  if (status != UCS_OK) {
    ucxx_error(
      "error on %s with status %d (%s)", operationName.c_str(), status, ucs_status_string(status));
    throw UCXXError(std::string("Error on ") + operationName + std::string(" message"));
  } else {
    ucxx_trace_req("%s completed immediately", operationName.c_str());
  }
}

}  // namespace ucxx
