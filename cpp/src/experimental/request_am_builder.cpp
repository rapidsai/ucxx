/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>
#include <variant>

#include <ucxx/constructors.h>
#include <ucxx/experimental/request_am_builder.h>

namespace ucxx {

namespace experimental {

RequestAmBuilder::RequestAmBuilder(std::shared_ptr<Endpoint> endpoint,
                                   std::variant<data::AmSend, data::AmReceive> requestData)
  : _endpoint(std::move(endpoint)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestAm> RequestAmBuilder::build() const
{
  return ucxx::createRequestAm(
    _endpoint, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
}

RequestAmBuilder::operator std::shared_ptr<RequestAm>() const
{
  return ucxx::createRequestAm(
    _endpoint, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
}

}  // namespace experimental

}  // namespace ucxx
