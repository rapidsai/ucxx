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
#include <ucxx/experimental/request_am_builder.h>
#include <ucxx/request.h>
#include <ucxx/request_am.h>

namespace ucxx {

namespace experimental {

RequestAmBuilder::RequestAmBuilder(std::shared_ptr<Endpoint> endpoint,
                                   std::variant<data::AmSend, data::AmReceive> requestData)
  : _endpoint(std::move(endpoint)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestAm> RequestAmBuilder::build()
{
  markBuilt();
  auto req = ucxx::createRequestAm(
    _endpoint, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
  detail::registerInflightRequest(_endpoint, req);
  return req;
}

RequestAmBuilder::operator std::shared_ptr<RequestAm>() { return build(); }

RequestAmBuilder::operator std::shared_ptr<Request>() { return build(); }

}  // namespace experimental

}  // namespace ucxx
