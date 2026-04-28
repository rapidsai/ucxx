/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>
#include <variant>

#include <ucxx/constructors.h>
#include <ucxx/experimental/request_tag_multi_builder.h>

namespace ucxx {

namespace experimental {

RequestTagMultiBuilder::RequestTagMultiBuilder(
  std::shared_ptr<Endpoint> endpoint,
  std::variant<data::TagMultiSend, data::TagMultiReceive> requestData)
  : _endpoint(std::move(endpoint)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestTagMulti> RequestTagMultiBuilder::build() const
{
  return ucxx::createRequestTagMulti(_endpoint, _requestData, _enablePythonFuture);
}

RequestTagMultiBuilder::operator std::shared_ptr<RequestTagMulti>() const
{
  return ucxx::createRequestTagMulti(_endpoint, _requestData, _enablePythonFuture);
}

}  // namespace experimental

}  // namespace ucxx
