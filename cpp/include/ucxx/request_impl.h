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
                         std::shared_ptr<NotificationRequest> notificationRequest,
                         const std::string operationName)
  : _endpoint{endpoint}, _notificationRequest(notificationRequest), _operationName(operationName)
{
  auto worker = UCXXEndpoint::getWorker(endpoint->getParent());

  if (worker == nullptr || worker->get_handle() == nullptr)
    throw ucxx::UCXXError("Worker not initialized");
  if (endpoint == nullptr || endpoint->getHandle() == nullptr)
    throw ucxx::UCXXError("Endpoint not initialized");

  _handle = std::make_shared<ucxx_request_t>();
#if UCXX_ENABLE_PYTHON
  _handle->py_future = worker->getPythonFuture();
  ucxx_trace_req("request->py_future: %p", _handle->py_future.get());
#endif

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

void UCXXRequest::process()
{
  // Operation completed immediately
  if (_requestStatusPtr == NULL) {
    _requestStatus = UCS_OK;
  } else {
    if (UCS_PTR_IS_ERR(_requestStatusPtr)) {
      _requestStatus = UCS_PTR_STATUS(_requestStatusPtr);
    } else if (UCS_PTR_IS_PTR(_requestStatusPtr)) {
      // Completion will be handled by callback
      _handle->request = _requestStatusPtr;
      return;
    } else {
      _requestStatus = UCS_OK;
    }
  }

  setStatus();

  ucxx_trace_req("_handle->callback: %p", _handle->callback.target<void (*)(void)>());
  if (_handle->callback) _handle->callback(_handle->callback_data);

  if (_requestStatus != UCS_OK) {
    ucxx_error("error on %s with status %d (%s)",
               _operationName.c_str(),
               _requestStatus,
               ucs_status_string(_requestStatus));
    throw UCXXError(std::string("Error on ") + _operationName + std::string(" message"));
  } else {
    ucxx_trace_req("%s completed immediately", _operationName.c_str());
  }
}

}  // namespace ucxx
