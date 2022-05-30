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
#include <ucxx/endpoint.h>
#include <ucxx/typedefs.h>

#if UCXX_ENABLE_PYTHON
#include <Python.h>
#endif

namespace ucxx {

Request::Request(std::shared_ptr<Endpoint> endpoint,
                 std::shared_ptr<NotificationRequest> notificationRequest,
                 const std::string operationName,
                 const bool enablePythonFuture)
  : _endpoint{endpoint},
    _notificationRequest(notificationRequest),
    _operationName(operationName),
    _enablePythonFuture(enablePythonFuture)
{
  auto worker = Endpoint::getWorker(endpoint->getParent());

  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");
  if (endpoint == nullptr || endpoint->getHandle() == nullptr)
    throw ucxx::Error("Endpoint not initialized");

  _handle = std::make_shared<ucxx_request_t>();
#if UCXX_ENABLE_PYTHON
  if (_enablePythonFuture) {
    _handle->py_future = worker->getPythonFuture();
    ucxx_trace_req("request->py_future: %p", _handle->py_future.get());
  }
#endif

  setParent(endpoint);
}

Request::~Request()
{
  if (_handle == nullptr) return;

  _endpoint->removeInflightRequest(this);
}

void Request::cancel()
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(getParent());
  auto worker   = std::dynamic_pointer_cast<Worker>(endpoint->getParent());
  ucp_request_cancel(worker->getHandle(), _handle->request);
}

std::shared_ptr<ucxx_request_t> Request::getHandle() { return _handle; }

ucs_status_t Request::getStatus() { return _handle->status; }

PyObject* Request::getPyFuture()
{
#if UCXX_ENABLE_PYTHON
  return (PyObject*)_handle->py_future->getHandle();
#else
  return NULL;
#endif
}

void Request::checkError()
{
  // Marking the pointer volatile is necessary to ensure the compiler
  // won't optimize the condition out when using a separate worker
  // progress thread
  volatile auto handle = _handle.get();
  switch (handle->status) {
    case UCS_OK:
    case UCS_INPROGRESS: return;
    case UCS_ERR_CANCELED: throw CanceledError(ucs_status_string(handle->status)); break;
    default: throw Error(ucs_status_string(handle->status)); break;
  }
}

template <typename Rep, typename Period>
bool Request::isCompleted(std::chrono::duration<Rep, Period> period)
{
  // Marking the pointer volatile is necessary to ensure the compiler
  // won't optimize the condition out when using a separate worker
  // progress thread
  volatile auto handle = _handle.get();
  return handle->status != UCS_INPROGRESS;
}

bool Request::isCompleted(int64_t periodNs)
{
  return isCompleted(std::chrono::nanoseconds(periodNs));
}

void Request::callback(void* request, ucs_status_t status)
{
  _requestStatus = ucp_request_check_status(request);

  if (_handle == nullptr)
    ucxx_error(
      "error when _callback was called for \"%s\", "
      "probably due to tag_msg() return value being deleted "
      "before completion.",
      _operationName.c_str());

  ucxx_trace_req("_calback called for \"%s\" with status %d (%s)",
                 _operationName.c_str(),
                 _requestStatus,
                 ucs_status_string(_requestStatus));

  _requestStatus = ucp_request_check_status(_requestStatusPtr);
  setStatus();

  ucxx_trace_req("_handle->callback: %p", _handle->callback.target<void (*)(void)>());
  if (_handle->callback) _handle->callback(_handle->callback_data);

  ucp_request_free(request);
}

void Request::process()
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
    throw Error(std::string("Error on ") + _operationName + std::string(" message"));
  } else {
    ucxx_trace_req("%s completed immediately", _operationName.c_str());
  }
}

void Request::setStatus()
{
  _handle->status = _requestStatus;

#if UCXX_ENABLE_PYTHON
  if (_enablePythonFuture) {
    auto future = std::static_pointer_cast<ucxx::python::Future>(_handle->py_future);
    future->notify(_requestStatus);
  }
#endif
}

}  // namespace ucxx
