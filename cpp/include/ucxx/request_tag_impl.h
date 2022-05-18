/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucxx/notification_request.h>
#include <ucxx/request_tag.h>

namespace ucxx {

std::shared_ptr<UCXXRequestTag> createRequestTag(
  std::shared_ptr<UCXXEndpoint> endpoint,
  bool send,
  void* buffer,
  size_t length,
  ucp_tag_t tag,
  std::function<void(std::shared_ptr<void>)> callbackFunction = nullptr,
  std::shared_ptr<void> callbackData                          = nullptr)
{
  return std::shared_ptr<UCXXRequestTag>(
    new UCXXRequestTag(endpoint, send, buffer, length, tag, callbackFunction, callbackData));
}

}  // namespace ucxx
