/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>
#include <variant>

#include <ucxx/constructors.h>
#include <ucxx/experimental/request_stream_builder.h>

namespace ucxx {

namespace experimental {

RequestStreamBuilder::RequestStreamBuilder(
  std::shared_ptr<Endpoint> endpoint,
  std::variant<data::StreamSend, data::StreamReceive> requestData)
  : _endpoint(std::move(endpoint)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestStream> RequestStreamBuilder::build() const
{
  return ucxx::createRequestStream(_endpoint, _requestData, _enablePythonFuture);
}

RequestStreamBuilder::operator std::shared_ptr<RequestStream>() const
{
  return ucxx::createRequestStream(_endpoint, _requestData, _enablePythonFuture);
}

}  // namespace experimental

}  // namespace ucxx
