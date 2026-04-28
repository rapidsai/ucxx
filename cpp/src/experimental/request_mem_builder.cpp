/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>
#include <variant>

#include <ucxx/constructors.h>
#include <ucxx/experimental/request_mem_builder.h>

namespace ucxx {

namespace experimental {

RequestMemBuilder::RequestMemBuilder(std::shared_ptr<Endpoint> endpoint,
                                     std::variant<data::MemPut, data::MemGet> requestData)
  : _endpoint(std::move(endpoint)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestMem> RequestMemBuilder::build() const
{
  return ucxx::createRequestMem(
    _endpoint, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
}

RequestMemBuilder::operator std::shared_ptr<RequestMem>() const
{
  return ucxx::createRequestMem(
    _endpoint, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
}

}  // namespace experimental

}  // namespace ucxx
