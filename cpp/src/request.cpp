/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <chrono>
#include <memory>
#include <sstream>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils/ucx.h>

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
  _worker =
    _endpoint ? _endpoint->getWorker() : std::dynamic_pointer_cast<Worker>(endpointOrWorker);

  if (_worker == nullptr || _worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");
  if (_endpoint != nullptr && _endpoint->getHandle() == nullptr)
    throw ucxx::Error("Endpoint not initialized");

  _enablePythonFuture &= _worker->isFutureEnabled();
  if (_enablePythonFuture) {
    _future = _worker->getFuture();
    ucxx_trace_req("req: %p, _future: %p", _request, _future.get());
  }

  std::stringstream ss;

  if (_endpoint) {
    setParent(_endpoint);
    ss << "ep " << _endpoint->getHandle();
  } else {
    setParent(_worker);
    ss << "worker " << _worker->getHandle();
  }

  _ownerString = ss.str();

  ucxx_trace("Request created: %p, %s", this, _operationName.c_str());
}

Request::~Request() { ucxx_trace("Request destroyed: %p, %s", this, _operationName.c_str()); }

void Request::cancel()
{
  if (_status == UCS_INPROGRESS) {
    if (UCS_PTR_IS_ERR(_request)) {
      ucs_status_t status = UCS_PTR_STATUS(_request);
      ucxx_trace_req_f(_ownerString.c_str(),
                       _request,
                       _operationName.c_str(),
                       "unprocessed request during cancelation contains error: %d (%s)",
                       status,
                       ucs_status_string(status));
    } else {
      ucxx_trace_req_f(_ownerString.c_str(), _request, _operationName.c_str(), "canceling");
      ucp_request_cancel(_worker->getHandle(), _request);
    }
  } else {
    auto status = _status.load();
    ucxx_trace_req_f(_ownerString.c_str(),
                     _request,
                     _operationName.c_str(),
                     "already completed with status: %d (%s)",
                     status,
                     ucs_status_string(status));
  }
}

ucs_status_t Request::getStatus() { return _status; }

void* Request::getFuture() { return _future ? _future->getHandle() : nullptr; }

void Request::checkError()
{
  // Only load the atomic variable once
  auto status = _status.load();

  utils::ucsErrorThrow(status, status == UCS_ERR_MESSAGE_TRUNCATED ? _status_msg : std::string());
}

bool Request::isCompleted() { return _status != UCS_INPROGRESS; }

void Request::callback(void* request, ucs_status_t status)
{
  ucxx_trace_req_f(_ownerString.c_str(),
                   request,
                   _operationName.c_str(),
                   "callback %p",
                   _callback.target<void (*)(void)>());
  if (_callback) _callback(status, _callbackData);

  ucp_request_free(request);

  ucxx_trace("Request completed: %p, handle: %p", this, request);
  setStatus(status);
}

void Request::process()
{
  ucs_status_t status = _status.load();

  if (UCS_PTR_IS_ERR(_request)) {
    // Operation errored immediately
    status = UCS_PTR_STATUS(_request);
  } else if (UCS_PTR_IS_PTR(_request)) {
    // Completion will be handled by callback
    ucxx_trace_req_f(_ownerString.c_str(),
                     _request,
                     _operationName.c_str(),
                     "completion will be handled by callback");
    ucxx_trace("Request submitted: %p, handle: %p", this, _request);
    return;
  } else {
    // Operation completed immediately
    status = UCS_OK;
  }

  ucxx_trace_req_f(_ownerString.c_str(),
                   _request,
                   _operationName.c_str(),
                   "status %d (%s)",
                   status,
                   ucs_status_string(status));

  if (status != UCS_OK) {
    ucxx_error(
      "error on %s with status %d (%s)", _operationName.c_str(), status, ucs_status_string(status));
  } else {
    ucxx_trace_req_f(
      _ownerString.c_str(), _request, _operationName.c_str(), "completed immediately");
  }

  ucxx_trace_req_f(_ownerString.c_str(),
                   _request,
                   _operationName.c_str(),
                   "callback %p",
                   _callback.target<void (*)(void)>());
  if (_callback) _callback(status, _callbackData);

  setStatus(status);
}

void Request::setStatus(ucs_status_t status)
{
  if (_endpoint != nullptr) _endpoint->removeInflightRequest(this);
  _worker->removeInflightRequest(this);

  if (_status == UCS_INPROGRESS) {
    // If the status is not `UCS_INPROGRESS`, the derived class has already set the
    // status, a truncated message for example.
    _status.store(status);
  }

  ucs_status_t s = _status;

  ucxx_trace_req_f(_ownerString.c_str(),
                   _request,
                   _operationName.c_str(),
                   "callback called with status %d (%s)",
                   s,
                   ucs_status_string(s));

  if (_enablePythonFuture) {
    auto future = std::static_pointer_cast<ucxx::Future>(_future);
    future->notify(status);
  }
}

const std::string& Request::getOwnerString() const { return _ownerString; }

}  // namespace ucxx
