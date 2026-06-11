/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <ucxx/constructors.h>
#include <ucxx/endpoint.h>
#include <ucxx/experimental/detail/register_inflight_request.h>
#include <ucxx/experimental/request_flush_builder.h>
#include <ucxx/request.h>
#include <ucxx/request_flush.h>
#include <ucxx/worker.h>

namespace ucxx {

namespace experimental {

RequestFlushBuilder::RequestFlushBuilder(std::shared_ptr<Component> endpointOrWorker,
                                         data::Flush requestData)
  : _endpointOrWorker(std::move(endpointOrWorker)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestFlush> RequestFlushBuilder::build() const
{
  markBuilt();
  auto req = ucxx::createRequestFlush(
    _endpointOrWorker, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
  detail::registerInflightRequest(_endpointOrWorker, req);
  return req;
}

RequestFlushBuilder::operator std::shared_ptr<RequestFlush>() const { return build(); }

RequestFlushBuilder::operator std::shared_ptr<Request>() const { return build(); }

}  // namespace experimental

}  // namespace ucxx
