/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>
#include <variant>

#include <ucxx/constructors.h>
#include <ucxx/endpoint.h>
#include <ucxx/experimental/request_tag_builder.h>
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

std::shared_ptr<RequestTag> RequestTagBuilder::build() const
{
  auto req = ucxx::createRequestTag(
    _endpointOrWorker, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
  if (auto ep = std::dynamic_pointer_cast<Endpoint>(_endpointOrWorker))
    (void)ep->registerInflightRequest(req);
  else if (auto wk = std::dynamic_pointer_cast<Worker>(_endpointOrWorker))
    (void)wk->registerInflightRequest(req);
  return req;
}

RequestTagBuilder::operator std::shared_ptr<RequestTag>() const { return build(); }

}  // namespace experimental

}  // namespace ucxx
