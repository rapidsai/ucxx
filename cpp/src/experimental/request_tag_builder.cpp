/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>
#include <variant>

#include <ucxx/constructors.h>
#include <ucxx/experimental/request_tag_builder.h>

namespace ucxx {

namespace experimental {

RequestTagBuilder::RequestTagBuilder(
  std::shared_ptr<Component> endpointOrWorker,
  std::variant<data::TagSend, data::TagReceive, data::TagReceiveWithHandle> requestData)
  : _endpointOrWorker(std::move(endpointOrWorker)), _requestData(std::move(requestData))
{
}

RequestTagBuilder& RequestTagBuilder::pythonFuture(bool enable)
{
  _enablePythonFuture = enable;
  return *this;
}

RequestTagBuilder& RequestTagBuilder::callbackFunction(RequestCallbackUserFunction fn)
{
  _callbackFunction = std::move(fn);
  return *this;
}

RequestTagBuilder& RequestTagBuilder::callbackData(RequestCallbackUserData data)
{
  _callbackData = std::move(data);
  return *this;
}

std::shared_ptr<RequestTag> RequestTagBuilder::build() const
{
  return ucxx::createRequestTag(
    _endpointOrWorker, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
}

RequestTagBuilder::operator std::shared_ptr<RequestTag>() const
{
  return ucxx::createRequestTag(
    _endpointOrWorker, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
}

}  // namespace experimental

}  // namespace ucxx
