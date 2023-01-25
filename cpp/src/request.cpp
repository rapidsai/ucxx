/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <chrono>
#include <memory>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils/ucx.h>

#if UCXX_ENABLE_PYTHON
#include <Python.h>
#endif

namespace ucxx {

Request::Request(std::shared_ptr<Component> endpointOrWorker,
                 std::shared_ptr<DelayedSubmission> delayedSubmission,
                 const std::string operationName,
                 const bool enablePythonFuture)
  : _delayedSubmission(delayedSubmission),
    _operationName(operationName),
    _enablePythonFuture(enablePythonFuture)
{
  _endpoint = std::dynamic_pointer_cast<Endpoint>(endpointOrWorker);
  _worker   = _endpoint ? Endpoint::getWorker(_endpoint->getParent())
                        : std::dynamic_pointer_cast<Worker>(endpointOrWorker);

  if (_worker == nullptr || _worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");
  if (_endpoint != nullptr && _endpoint->getHandle() == nullptr)
    throw ucxx::Error("Endpoint not initialized");

#if UCXX_ENABLE_PYTHON
  _enablePythonFuture &= _worker->isPythonFutureEnabled();
  if (_enablePythonFuture) {
    _pythonFuture = _worker->getPythonFuture();
    ucxx_trace_req("req: %p, _pythonFuture: %p", _request, _pythonFuture.get());
  }
#endif

  if (_endpoint)
    setParent(_endpoint);
  else
    setParent(_worker);
}

Request::~Request()
{
  if (_endpoint)
    _endpoint->removeInflightRequest(this);
  else
    _worker->removeInflightRequest(this);
}

void Request::cancel()
{
  if (_request == nullptr) {
    ucxx_trace_req("req: %p already completed or cancelled", _request);
  } else if (_status == UCS_INPROGRESS) {
    // Errored requests cannot be canceled
    if (UCS_PTR_IS_ERR(_request)) {
      ucs_status_t status = UCS_PTR_STATUS(_request);
      ucxx_trace_req("req: requested was not processed, contains error: %d (%s)",
                     status,
                     ucs_status_string(status));
    } else {
      ucxx_trace_req("req: %p, cancelling", _request);
      ucp_request_cancel(_worker->getHandle(), _request);
    }
  }
}

ucs_status_t Request::getStatus() { return _status; }

PyObject* Request::getPyFuture()
{
#if UCXX_ENABLE_PYTHON
  return (PyObject*)_pythonFuture->getHandle();
#else
  return nullptr;
#endif
}

void Request::checkError()
{
  // Only load the atomic variable once
  auto status = _status.load();

  utils::ucsErrorThrow(status, status == UCS_ERR_MESSAGE_TRUNCATED ? _status_msg : std::string());
}

bool Request::isCompleted() { return _status != UCS_INPROGRESS; }

void Request::callback(void* request, ucs_status_t status)
{
  setStatus(status);

  ucxx_trace_req("req: %p, _callback: %p", request, _callback.target<void (*)(void)>());
  if (_callback) _callback(_callbackData);

  ucp_request_free(request);
}

void Request::process()
{
  ucs_status_t status = _status.load();

  // Operation completed immediately
  if (_request == nullptr) {
    status = UCS_OK;
  } else {
    if (UCS_PTR_IS_ERR(_request)) {
      status = UCS_PTR_STATUS(_request);
    } else if (UCS_PTR_IS_PTR(_request)) {
      // Completion will be handled by callback
      return;
    } else {
      status = UCS_OK;
    }
  }

  ucxx_trace_req("req: %p, status: %d (%s)", _request, status, ucs_status_string(status));

  ucxx_trace_req("req: %p, callback: %p", _request, _callback.target<void (*)(void)>());
  if (_callback) _callback(_callbackData);

  if (status != UCS_OK) {
    ucxx_error(
      "error on %s with status %d (%s)", _operationName.c_str(), status, ucs_status_string(status));
  } else {
    ucxx_trace_req("req: %p, %s completed immediately", _request, _operationName.c_str());
  }

  setStatus(status);
}

void Request::setStatus(ucs_status_t status)
{
  if (_status == UCS_INPROGRESS) {
    // If the status is not `UCS_INPROGRESS`, the derived class has already set the
    // status, a truncated message for example.
    _status.store(status);
  }

  ucs_status_t s = _status;

  ucxx_trace_req("req: %p, callback called \"%s\" with status %d (%s)",
                 _request,
                 _operationName.c_str(),
                 s,
                 ucs_status_string(s));

#if UCXX_ENABLE_PYTHON
  if (_enablePythonFuture) {
    auto future = std::static_pointer_cast<ucxx::python::Future>(_pythonFuture);
    future->notify(status);
  }
#endif
}

}  // namespace ucxx
