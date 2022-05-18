/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <chrono>
#include <memory>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint_impl.h>
#include <ucxx/typedefs.h>

#if UCXX_ENABLE_PYTHON
#include <Python.h>
#endif

namespace ucxx {

UCXXRequest::UCXXRequest(std::shared_ptr<UCXXEndpoint> endpoint,
                         std::shared_ptr<ucxx_request_t> request,
                         std::shared_ptr<NotificationRequest> notificationRequest)
  : _handle{request}, _endpoint{endpoint}, _notificationRequest(notificationRequest)
{
  if (endpoint == nullptr || endpoint->getHandle() == nullptr)
    throw ucxx::UCXXError("Endpoint not initialized");

  setParent(endpoint);
}

UCXXRequest::~UCXXRequest()
{
  if (_handle == nullptr) return;

  _endpoint->removeInflightRequest(this);
}

void UCXXRequest::cancel()
{
  auto endpoint = std::dynamic_pointer_cast<UCXXEndpoint>(getParent());
  auto worker   = std::dynamic_pointer_cast<UCXXWorker>(endpoint->getParent());
  ucp_request_cancel(worker->get_handle(), _handle->request);
}

std::shared_ptr<ucxx_request_t> UCXXRequest::getHandle() { return _handle; }

ucs_status_t UCXXRequest::getStatus() { return _handle->status; }

PyObject* UCXXRequest::getPyFuture()
{
#if UCXX_ENABLE_PYTHON
  return (PyObject*)_handle->py_future->getHandle();
#else
  return NULL;
#endif
}

void UCXXRequest::checkError()
{
  // Marking the pointer volatile is necessary to ensure the compiler
  // won't optimize the condition out when using a separate worker
  // progress thread
  volatile auto handle = _handle.get();
  switch (handle->status) {
    case UCS_OK:
    case UCS_INPROGRESS: return;
    case UCS_ERR_CANCELED: throw UCXXCanceledError(ucs_status_string(handle->status)); break;
    default: throw UCXXError(ucs_status_string(handle->status)); break;
  }
}

template <typename Rep, typename Period>
bool UCXXRequest::isCompleted(std::chrono::duration<Rep, Period> period)
{
  // Marking the pointer volatile is necessary to ensure the compiler
  // won't optimize the condition out when using a separate worker
  // progress thread
  volatile auto handle = _handle.get();
  return handle->status != UCS_INPROGRESS;
}

bool UCXXRequest::isCompleted(int64_t periodNs)
{
  return isCompleted(std::chrono::nanoseconds(periodNs));
}

void UCXXRequest::process(ucp_worker_h worker, void* request, std::string operationName)
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
      _handle->request = request;
      return;
    } else {
      status = UCS_OK;
    }
  }

  setStatus(status);

  ucxx_trace_req("_handle->callback: %p", _handle->callback.target<void (*)(void)>());
  if (_handle->callback) _handle->callback(_handle->callback_data);

  if (status != UCS_OK) {
    ucxx_error(
      "error on %s with status %d (%s)", operationName.c_str(), status, ucs_status_string(status));
    throw UCXXError(std::string("Error on ") + operationName + std::string(" message"));
  } else {
    ucxx_trace_req("%s completed immediately", operationName.c_str());
  }
}

}  // namespace ucxx
