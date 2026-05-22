/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <ucxx/constructors.h>
#include <ucxx/endpoint.h>
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
  auto req = ucxx::createRequestEndpointClose(
    _endpoint, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
  (void)_endpoint->registerInflightRequest(req);
  return req;
}

RequestEndpointCloseBuilder::operator std::shared_ptr<RequestEndpointClose>() const
{
  return build();
}

RequestEndpointCloseBuilder::operator std::shared_ptr<Request>() const { return build(); }

}  // namespace experimental

}  // namespace ucxx
