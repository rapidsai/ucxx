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
                 std::shared_ptr<DelayedSubmission> delayedSubmission,
                 const std::string operationName,
                 const bool enablePythonFuture)
  : _endpoint{endpoint},
    _delayedSubmission(delayedSubmission),
    _operationName(operationName),
    _enablePythonFuture(enablePythonFuture)
{
  auto worker = Endpoint::getWorker(endpoint->getParent());

  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");
  if (endpoint == nullptr || endpoint->getHandle() == nullptr)
    throw ucxx::Error("Endpoint not initialized");

#if UCXX_ENABLE_PYTHON
  _enablePythonFuture &= worker->isPythonFutureEnabled();
  if (_enablePythonFuture) {
    _pythonFuture = worker->getPythonFuture();
    ucxx_trace_req("request->py_future: %p", _pythonFuture.get());
  }
#endif

  setParent(endpoint);
}

Request::~Request()
{
  if (_request == nullptr) return;

  _endpoint->removeInflightRequest(this);
}

void Request::cancel()
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(getParent());
  auto worker   = std::dynamic_pointer_cast<Worker>(endpoint->getParent());
  ucp_request_cancel(worker->getHandle(), _request);
}

ucs_status_t Request::getStatus() { return _status; }

PyObject* Request::getPyFuture()
{
#if UCXX_ENABLE_PYTHON
  return (PyObject*)_pythonFuture->getHandle();
#else
  return NULL;
#endif
}

void Request::checkError()
{
  // Marking the pointer volatile is necessary to ensure the compiler
  // won't optimize the condition out when using a separate worker
  // progress thread
  switch (_status) {
    case UCS_OK:
    case UCS_INPROGRESS: return;
    case UCS_ERR_CANCELED: throw CanceledError(ucs_status_string(_status)); break;
    case UCS_ERR_MESSAGE_TRUNCATED: throw MessageTruncatedError(_status_msg); break;
    default: throw Error(ucs_status_string(_status)); break;
  }
}

template <typename Rep, typename Period>
bool Request::isCompleted(std::chrono::duration<Rep, Period> period)
{
  // Marking the pointer volatile is necessary to ensure the compiler
  // won't optimize the condition out when using a separate worker
  // progress thread
  return _status != UCS_INPROGRESS;
}

bool Request::isCompleted(int64_t periodNs)
{
  return isCompleted(std::chrono::nanoseconds(periodNs));
}

void Request::callback(void* request, ucs_status_t status)
{
  ucs_status_t s;

  if (_status == UCS_INPROGRESS) {
    s       = ucp_request_check_status(request);
    _status = s;
  } else {
    // Derived class has already set the status, e.g., a truncated message.
    s = _status;
  }

  ucxx_trace_req("Request::callback called for \"%s\" with status %d (%s)",
                 _operationName.c_str(),
                 s,
                 ucs_status_string(s));

  ucxx_trace_req("_callback: %p", _callback.target<void (*)(void)>());
  if (_callback) _callback(_callbackData);

  ucp_request_free(request);

  setStatus();
}

void Request::process()
{
  // Operation completed immediately
  if (_request == NULL) {
    _status = UCS_OK;
  } else {
    if (UCS_PTR_IS_ERR(_request)) {
      _status = UCS_PTR_STATUS(_request);
    } else if (UCS_PTR_IS_PTR(_request)) {
      // Completion will be handled by callback
      return;
    } else {
      _status = UCS_OK;
    }
  }

  ucs_status_t status = _status.load();
  ucxx_trace_req("status: %d (%s)", status, ucs_status_string(status));

  ucxx_trace_req("callback: %p", _callback.target<void (*)(void)>());
  if (_callback) _callback(_callbackData);

  if (status != UCS_OK) {
    ucxx_error(
      "error on %s with status %d (%s)", _operationName.c_str(), status, ucs_status_string(status));
  } else {
    ucxx_trace_req("%s completed immediately", _operationName.c_str());
  }

  setStatus();
}

void Request::setStatus()
{
#if UCXX_ENABLE_PYTHON
  if (_enablePythonFuture) {
    auto future = std::static_pointer_cast<ucxx::python::Future>(_pythonFuture);
    future->notify(_status);
  }
#endif
}

}  // namespace ucxx
