/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
  ucxx_trace("ucxx::Request destroyed (%s): %p", _operationName.c_str(), this);
}

void Request::cancel()
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
      ucxx_trace_req_f(_ownerString.c_str(), this, _request, _operationName.c_str(), "canceling");
      if (_request != nullptr) ucp_request_cancel(_worker->getHandle(), _request);
    }
  } else {
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "already completed with status: %d (%s)",
                     _status,
                     ucs_status_string(_status));
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

  if (UCS_PTR_IS_PTR(_request)) ucp_request_free(request);

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

    if (_endpoint != nullptr) _endpoint->removeInflightRequest(this);
    _worker->removeInflightRequest(this);

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

void Request::queryRequestAttributes()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);

  if (_isRequestAttrValid) return;

  ucp_request_attr_t result;

  // Get the debug string size from worker attributes
  auto worker_attr = _worker->queryAttributes();

  // Allocate buffer for debug string with size from worker attributes
  std::vector<char> debug_str(worker_attr.max_debug_string, '\0');

  result.field_mask = UCP_REQUEST_ATTR_FIELD_STATUS |           // Request status
                      UCP_REQUEST_ATTR_FIELD_MEM_TYPE |         // Memory type
                      UCP_REQUEST_ATTR_FIELD_INFO_STRING |      // Debug string
                      UCP_REQUEST_ATTR_FIELD_INFO_STRING_SIZE;  // Debug string size

  // Set up the debug string buffer
  result.debug_string      = debug_str.data();
  result.debug_string_size = debug_str.size();

  if (UCS_PTR_IS_PTR(_request)) {
    result.status = ucp_request_query(_request, &result);
    if (result.status == UCS_OK && result.debug_string != nullptr) {
      _requestAttr.debugString = std::string(result.debug_string);
      _requestAttr.memoryType  = result.mem_type;
      _requestAttr.status      = result.status;
      _isRequestAttrValid      = true;
    }
  }
}

Request::RequestAttributes Request::getRequestAttributes()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);

  if (_isRequestAttrValid)
    return _requestAttr;
  else
    throw ucxx::Error("Request attributes not available yet");
}

std::shared_ptr<Buffer> Request::getRecvBuffer() { return nullptr; }

}  // namespace ucxx
