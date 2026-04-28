/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <ucxx/constructors.h>
#include <ucxx/experimental/request_endpoint_close_builder.h>

namespace ucxx {

namespace experimental {

RequestEndpointCloseBuilder::RequestEndpointCloseBuilder(std::shared_ptr<Endpoint> endpoint,
                                                         data::EndpointClose requestData)
  : _endpoint(std::move(endpoint)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestEndpointClose> RequestEndpointCloseBuilder::build() const
{
  return ucxx::createRequestEndpointClose(
    _endpoint, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
}

RequestEndpointCloseBuilder::operator std::shared_ptr<RequestEndpointClose>() const
{
  return ucxx::createRequestEndpointClose(
    _endpoint, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
}

}  // namespace experimental

}  // namespace ucxx
