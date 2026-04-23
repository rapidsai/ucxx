/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <ucxx/constructors.h>
#include <ucxx/experimental/request_flush_builder.h>

namespace ucxx {

namespace experimental {

RequestFlushBuilder::RequestFlushBuilder(std::shared_ptr<Component> endpointOrWorker,
                                         data::Flush requestData)
  : _endpointOrWorker(std::move(endpointOrWorker)), _requestData(std::move(requestData))
{
}

RequestFlushBuilder& RequestFlushBuilder::pythonFuture(bool enable)
{
  _enablePythonFuture = enable;
  return *this;
}

RequestFlushBuilder& RequestFlushBuilder::callbackFunction(RequestCallbackUserFunction fn)
{
  _callbackFunction = std::move(fn);
  return *this;
}

RequestFlushBuilder& RequestFlushBuilder::callbackData(RequestCallbackUserData data)
{
  _callbackData = std::move(data);
  return *this;
}

std::shared_ptr<RequestFlush> RequestFlushBuilder::build() const
{
  return ucxx::createRequestFlush(
    _endpointOrWorker, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
}

RequestFlushBuilder::operator std::shared_ptr<RequestFlush>() const
{
  return ucxx::createRequestFlush(
    _endpointOrWorker, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
}

}  // namespace experimental

}  // namespace ucxx
