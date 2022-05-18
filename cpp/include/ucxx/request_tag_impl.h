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
  std::shared_ptr<UCXXWorker> worker,
  std::shared_ptr<UCXXEndpoint> endpoint,
  bool send,
  void* buffer,
  size_t length,
  ucp_tag_t tag,
  std::function<void(std::shared_ptr<void>)> callbackFunction = nullptr,
  std::shared_ptr<void> callbackData                          = nullptr)
{
  auto request = std::shared_ptr<UCXXRequestTag>(
    new UCXXRequestTag(worker, endpoint, callbackFunction, callbackData));

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  auto notificationRequest = std::make_shared<NotificationRequest>(
    worker->get_handle(), endpoint->getHandle(), request->getHandle(), send, buffer, length, tag);
  worker->registerNotificationRequest(
    std::bind(
      std::mem_fn(&UCXXRequest::populateNotificationRequest), request, std::placeholders::_1),
    notificationRequest);

  return request;
}

}  // namespace ucxx
