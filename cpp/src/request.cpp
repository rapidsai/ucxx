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
                 const data::RequestData requestData,
                 const std::string operationName,
                 const bool enablePythonFuture,
                 RequestCallbackUserFunction callbackFunction,
                 RequestCallbackUserData callbackData)
  : _requestData(requestData),
    _operationName(operationName),
    _enablePythonFuture(enablePythonFuture),
    _callback(callbackFunction),
    _callbackData(callbackData)
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
    ucxx_trace_req_f(
      _ownerString.c_str(), this, _request, _operationName.c_str(), "future: %p", _future.get());
  }

  std::stringstream ss;

  if (_endpoint) {
    setParent(_endpoint);
    ss << "ucxx::Endpoint: " << _endpoint->getHandle();
    ucxx_trace("ucxx::Request created (%s): %p on ucxx::Endpoint: %p",
               _operationName.c_str(),
               this,
               _endpoint.get());
  } else {
    setParent(_worker);
    ss << "ucxx::Worker: " << _worker->getHandle();
    ucxx_trace("ucxx::Request created (%s): %p on ucxx::Worker: %p",
               _operationName.c_str(),
               this,
               _worker.get());
  }

  _ownerString = ss.str();
}

Request::~Request()
{
  if (_cancelCallback != nullptr) {
    auto completed  = _cancelCallbackNotifier.wait(1000000000 /* 1s */);
    _cancelCallback = nullptr;
  }
  if (UCS_PTR_IS_PTR(_request)) {
    ucxx_warn("ucxx::Request (%s) freeing: %p", _operationName.c_str(), _request);
    ucp_request_free(_request);
  }
  ucxx_trace("ucxx::Request destroyed (%s): %p", _operationName.c_str(), this);
}

void Request::removeInflightRequest()
{
  if (_endpoint != nullptr) _endpoint->removeInflightRequest(this);
  _worker->removeInflightRequest(this);
}

void Request::cancelImpl()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  if (_status == UCS_INPROGRESS) {
    if (UCS_PTR_IS_ERR(_request)) {
      ucs_status_t status = UCS_PTR_STATUS(_request);
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "unprocessed request during cancelation contains error: %d (%s)",
                       status,
                       ucs_status_string(status));
    } else {
      if (_request != nullptr) {
        ucs_status_t status = UCS_PTR_STATUS(_request);
        ucxx_trace_req_f(_ownerString.c_str(), this, _request, _operationName.c_str(), "canceling");
        ucp_request_cancel(_worker->getHandle(), _request);
        status = UCS_PTR_STATUS(_request);

        /**
         * Tag send requests cannot be canceled: https://github.com/openucx/ucx/issues/1162
         * This can be problematic for unmatched rendezvous tag send messages as it would
         * otherwise not complete cancelation, so we forcefully "cancel" the requests which
         * ultimately leads it reaching `ucp_request_free`. This currently causes the UCX
         * warnings "was not returned to mpool ucp_requests", which is likely a UCX bug.
         */
        if (_operationName == "tagSend") { setStatus(UCS_ERR_CANCELED); }
      }
    }
  } else {
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "already completed with status: %d (%s)",
                     _status,
                     ucs_status_string(_status));

    /**
     * Ensure the request is removed from the parent in case it got re-registered while it
     * was completing.
     */
    removeInflightRequest();
  }

  _cancelCallback = nullptr;
}

void Request::cancel()
{
  if (_worker->isProgressThreadRunning()) {
    _cancelCallback = [this]() {
      /**
       * FIXME: Check the callback hasn't run and object hasn't been destroyed. Long-term
       * fix is to allow deregistering generic callbacks with the worker.
       */
      if (_cancelCallback == nullptr) return;

      cancelImpl();
      _cancelCallbackNotifier.set();
    };
    // The Cancel callback is store in an attributed, thus we do not need to
    // cancel it if it fails to run immediately.
    std::ignore = _worker->registerGenericPre(_cancelCallback);
  } else {
    cancelImpl();
  }
}

ucs_status_t Request::getStatus()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  return _status;
}

void* Request::getFuture()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  return _future ? _future->getHandle() : nullptr;
}

void Request::checkError()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);

  utils::ucsErrorThrow(_status, _status == UCS_ERR_MESSAGE_TRUNCATED ? _status_msg : std::string());
}

bool Request::isCompleted()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  return _status != UCS_INPROGRESS;
}

void Request::callback(void* request, ucs_status_t status)
{
  /**
   * Prevent reference count to self from going to zero and thus cause self to be destroyed
   * while `callback()` executes.
   */
  decltype(shared_from_this()) selfReference = nullptr;
  try {
    selfReference = shared_from_this();
  } catch (std::bad_weak_ptr& exception) {
    ucxx_debug("ucxx::Request: %p destroyed before callback() was executed", this);
    return;
  }
  if (_status != UCS_INPROGRESS)
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "has status already set to %d (%s), callback setting %d (%s)",
                     _status,
                     ucs_status_string(_status),
                     status,
                     ucs_status_string(status));

  if (UCS_PTR_IS_PTR(_request)) {
    ucp_request_free(request);
    _request = nullptr;
  }

  ucxx_trace_req_f(_ownerString.c_str(), this, _request, _operationName.c_str(), "completed");
  setStatus(status);
  ucxx_trace_req_f(
    _ownerString.c_str(), this, _request, _operationName.c_str(), "isCompleted: %d", isCompleted());
}

void Request::process()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);

  ucs_status_t status = UCS_INPROGRESS;

  if (UCS_PTR_IS_ERR(_request)) {
    // Operation errored immediately
    status = UCS_PTR_STATUS(_request);
  } else if (UCS_PTR_IS_PTR(_request)) {
    // Completion will be handled by callback
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "completion will be handled by callback");
    return;
  } else {
    // Operation completed immediately
    status = UCS_OK;
  }

  ucxx_trace_req_f(_ownerString.c_str(),
                   this,
                   _request,
                   _operationName.c_str(),
                   "status %d (%s)",
                   status,
                   ucs_status_string(status));

  if (status != UCS_OK) {
    ucxx_debug(
      "error on %s with status %d (%s)", _operationName.c_str(), status, ucs_status_string(status));
  } else {
    ucxx_trace_req_f(
      _ownerString.c_str(), this, _request, _operationName.c_str(), "completed immediately");
  }

  setStatus(status);
}

void Request::setStatus(ucs_status_t status)
{
  {
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    removeInflightRequest();

    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "setStatus called with status %d (%s)",
                     status,
                     ucs_status_string(status));

    if (_status != UCS_INPROGRESS)
      ucxx_error(
        "ucxx::Request: %p, setStatus called with status: %d (%s) but status: %d (%s) was "
        "already set",
        this,
        status,
        ucs_status_string(status),
        _status,
        ucs_status_string(_status));
    _status = status;

    if (_enablePythonFuture) {
      auto future = std::static_pointer_cast<ucxx::Future>(_future);
      future->notify(status);
    }

    if (_callback) {
      ucxx_trace_req_f(
        _ownerString.c_str(), this, _request, _operationName.c_str(), "invoking user callback");
      _callback(status, _callbackData);
    }
  }
}

const std::string& Request::getOwnerString() const { return _ownerString; }

std::shared_ptr<Buffer> Request::getRecvBuffer() { return nullptr; }

}  // namespace ucxx
