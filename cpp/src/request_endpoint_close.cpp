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
#include <ucxx/request_endpoint_close.h>

namespace ucxx {

std::shared_ptr<RequestEndpointClose> createRequestEndpointClose(
  std::shared_ptr<Endpoint> endpoint,
  const std::variant<data::EndpointClose> requestData,
  const bool enablePythonFuture                = false,
  RequestCallbackUserFunction callbackFunction = nullptr,
  RequestCallbackUserData callbackData         = nullptr)
{
  std::shared_ptr<RequestEndpointClose> req = std::visit(
    data::dispatch{
      [&endpoint, &enablePythonFuture, &callbackFunction, &callbackData](
        data::EndpointClose endpointClose) {
        return std::shared_ptr<RequestEndpointClose>(new RequestEndpointClose(endpoint,
                                                                              endpointClose,
                                                                              "endpointClose",
                                                                              enablePythonFuture,
                                                                              callbackFunction,
                                                                              callbackData));
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

RequestEndpointClose::RequestEndpointClose(std::shared_ptr<Endpoint> endpoint,
                                           const std::variant<data::EndpointClose> requestData,
                                           const std::string operationName,
                                           const bool enablePythonFuture,
                                           RequestCallbackUserFunction callbackFunction,
                                           RequestCallbackUserData callbackData)
  : Request(endpoint, data::getRequestData(requestData), operationName, enablePythonFuture)
{
  std::visit(
    data::dispatch{
      [this](data::EndpointClose) {
        if (_endpoint == nullptr && _worker == nullptr)
          throw ucxx::Error("A valid endpoint or worker is required for a close operation.");
      },
      [](auto) { throw std::runtime_error("Unreachable"); },
    },
    requestData);

  _callback     = callbackFunction;
  _callbackData = callbackData;
}

void RequestEndpointClose::endpointCloseCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(
    req->getOwnerString().c_str(), nullptr, request, "endpointClose", "endpointCloseCallback");
  return req->callback(request, status);
}

void RequestEndpointClose::request()
{
  ucp_request_param_t param = {
    .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA, .user_data = this};

  void* request = nullptr;

  std::visit(
    data::dispatch{
      [this, &request, &param](data::EndpointClose) {
        param.cb.send = endpointCloseCallback;
        if (_endpoint != nullptr)
          request = ucp_ep_close_nbx(_endpoint->getHandle(), &param);
        else
          throw ucxx::Error("A valid endpoint or worker is required for a close operation.");
      },
      [](auto) { throw std::runtime_error("Unreachable"); },
    },
    _requestData);

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  _request = request;
}

static void logPopulateDelayedSubmission() {}

void RequestEndpointClose::populateDelayedSubmission()
{
  bool terminate =
    std::visit(data::dispatch{
                 [this](data::EndpointClose endpointClose) {
                   if (_endpoint != nullptr && _endpoint->getHandle() == nullptr) {
                     ucxx_warn("Endpoint is already closed");
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
    if (_enablePythonFuture)
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "populateDelayedSubmission, endpoint close, future: %p, future handle: %p",
                       _future.get(),
                       _future->getHandle());
    else
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "populateDelayedSubmission, endpoint close");
  };

  std::visit(data::dispatch{
               [this, &log](data::EndpointClose endpointClose) { log(); },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _requestData);

  process();
}

}  // namespace ucxx
