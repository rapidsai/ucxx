/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>
#include <variant>

#include <ucxx/constructors.h>
#include <ucxx/endpoint.h>
#include <ucxx/experimental/detail/register_inflight_request.h>
#include <ucxx/experimental/request_tag_builder.h>
#include <ucxx/request.h>
#include <ucxx/request_tag.h>
#include <ucxx/worker.h>

namespace ucxx {

namespace experimental {

RequestTagBuilder::RequestTagBuilder(
  std::shared_ptr<Component> endpointOrWorker,
  std::variant<data::TagSend, data::TagReceive, data::TagReceiveWithHandle> requestData)
  : _endpointOrWorker(std::move(endpointOrWorker)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestTag> RequestTagBuilder::build()
{
  markBuilt();
  auto req = ucxx::createRequestTag(
    _endpointOrWorker, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
  detail::registerInflightRequest(_endpointOrWorker, req);
  return req;
}

RequestTagBuilder::operator std::shared_ptr<RequestTag>() { return build(); }

RequestTagBuilder::operator std::shared_ptr<Request>() { return build(); }

}  // namespace experimental

}  // namespace ucxx
