/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucxx/notification_request.h>
#include <ucxx/request_stream.h>

namespace ucxx {

std::shared_ptr<UCXXRequestStream> createRequestStream(std::shared_ptr<UCXXWorker> worker,
                                                       std::shared_ptr<UCXXEndpoint> endpoint,
                                                       bool send,
                                                       void* buffer,
                                                       size_t length)
{
  auto request = std::shared_ptr<UCXXRequestStream>(new UCXXRequestStream(worker, endpoint));

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  auto notificationRequest = std::make_shared<NotificationRequest>(
    worker->get_handle(), endpoint->getHandle(), request->getHandle(), send, buffer, length);
  worker->registerNotificationRequest(
    std::bind(
      std::mem_fn(&UCXXRequest::populateNotificationRequest), request, std::placeholders::_1),
    notificationRequest);

  return request;
}

}  // namespace ucxx
