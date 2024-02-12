/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/buffer.h>
#include <ucxx/delayed_submission.h>
#include <ucxx/request_flush.h>

namespace ucxx {

std::shared_ptr<RequestFlush> createRequestFlush(
  std::shared_ptr<Component> endpointOrWorker,
  const std::variant<data::Flush> requestData,
  const bool enablePythonFuture                = false,
  RequestCallbackUserFunction callbackFunction = nullptr,
  RequestCallbackUserData callbackData         = nullptr)
{
  std::shared_ptr<RequestFlush> req = std::visit(
    data::dispatch{
      [&endpointOrWorker, &enablePythonFuture, &callbackFunction, &callbackData](
        data::Flush flush) {
        return std::shared_ptr<RequestFlush>(new RequestFlush(
          endpointOrWorker, flush, "flush", enablePythonFuture, callbackFunction, callbackData));
      },
    },
    requestData);

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  req->_worker->registerDelayedSubmission(
    req, std::bind(std::mem_fn(&Request::populateDelayedSubmission), req.get()));

  return req;
}

RequestFlush::RequestFlush(std::shared_ptr<Component> endpointOrWorker,
                           const std::variant<data::Flush> requestData,
                           const std::string operationName,
                           const bool enablePythonFuture,
                           RequestCallbackUserFunction callbackFunction,
                           RequestCallbackUserData callbackData)
  : Request(endpointOrWorker, data::getRequestData(requestData), operationName, enablePythonFuture)
{
  std::visit(
    data::dispatch{
      [this](data::Flush) {
        if (_endpoint == nullptr && _worker == nullptr)
          throw ucxx::Error("A valid endpoint or worker is required for a flush operation.");
      },
      [](auto) { throw std::runtime_error("Unreachable"); },
    },
    requestData);

  _callback     = callbackFunction;
  _callbackData = callbackData;
}

void RequestFlush::flushCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), nullptr, request, "flush", "flushCallback");
  return req->callback(request, status);
}

void RequestFlush::request()
{
  ucp_request_param_t param = {
    .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA, .user_data = this};

  void* request = nullptr;

  std::visit(
    data::dispatch{
      [this, &request, &param](data::Flush) {
        param.cb.send = flushCallback;
        if (_endpoint != nullptr)
          request = ucp_ep_flush_nbx(_endpoint->getHandle(), &param);
        else if (_worker != nullptr)
          request = ucp_worker_flush_nbx(_worker->getHandle(), &param);
        else
          throw ucxx::Error("A valid endpoint or worker is required for a flush operation.");
      },
      [](auto) { throw std::runtime_error("Unreachable"); },
    },
    _requestData);

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  _request = request;
}

static void logPopulateDelayedSubmission() {}

void RequestFlush::populateDelayedSubmission()
{
  bool terminate =
    std::visit(data::dispatch{
                 [this](data::Flush flush) {
                   if (_endpoint != nullptr && _endpoint->getHandle() == nullptr) {
                     ucxx_warn("Endpoint was closed before it could be flushed");
                     Request::callback(this, UCS_ERR_CANCELED);
                     return true;
                   } else if (_worker != nullptr && _worker->getHandle() == nullptr) {
                     ucxx_warn("Worker was closed before it could be flushed");
                     Request::callback(this, UCS_ERR_CANCELED);
                     return true;
                   }
                   return false;
                 },
                 [](auto) -> decltype(terminate) { throw std::runtime_error("Unreachable"); },
               },
               _requestData);
  if (terminate) return;

  request();

  auto log = [this]() {
    std::string flushComponent = "unknown";
    if (_endpoint != nullptr)
      flushComponent = "endpoint";
    else if (_worker != nullptr)
      flushComponent = "worker";

    if (_enablePythonFuture)
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "populateDelayedSubmission, flush (%s), future: %p, future handle: %p",
                       flushComponent.c_str(),
                       _future.get(),
                       _future->getHandle());
    else
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "populateDelayedSubmission, flush (%s)",
                       flushComponent.c_str());
  };

  std::visit(data::dispatch{
               [this, &log](data::Flush flush) { log(); },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _requestData);

  process();
}

}  // namespace ucxx
