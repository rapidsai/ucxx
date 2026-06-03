/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <ucxx/endpoint.h>
#include <ucxx/exception.h>
#include <ucxx/experimental/request_endpoint_close_builder.h>
#include <ucxx/request.h>
#include <ucxx/request_endpoint_close.h>

namespace ucxx {

namespace experimental {

RequestEndpointCloseBuilder::RequestEndpointCloseBuilder(std::shared_ptr<Endpoint> endpoint,
                                                         data::EndpointClose requestData)
  : _endpoint(std::move(endpoint)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestEndpointClose> RequestEndpointCloseBuilder::build() const
{
  if (_endpoint == nullptr)
    throw ucxx::Error("A valid endpoint is required for a close operation.");

  return _endpoint->closeRequest(
    _requestData._force, _enablePythonFuture, _callbackFunction, _callbackData);
}

RequestEndpointCloseBuilder::operator std::shared_ptr<RequestEndpointClose>() const
{
  return build();
}

RequestEndpointCloseBuilder::operator std::shared_ptr<Request>() const { return build(); }

}  // namespace experimental

}  // namespace ucxx
