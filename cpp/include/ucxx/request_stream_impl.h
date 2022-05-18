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
  auto request = std::make_shared<ucxx_request_t>();
#if UCXX_ENABLE_PYTHON
  request->py_future = worker->getPythonFuture();
  ucxx_trace_req("request: %p, request->py_future: %p", request.get(), request->py_future.get());
#endif
  request->callback      = nullptr;
  request->callback_data = nullptr;

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  auto notificationRequest = std::make_shared<NotificationRequest>(
    worker->get_handle(), endpoint->getHandle(), request, send, buffer, length);
  worker->registerNotificationRequest(UCXXRequestStream::populateNotificationRequestStream,
                                      notificationRequest);

  return std::shared_ptr<UCXXRequestStream>(
    new UCXXRequestStream(endpoint, endpoint->_inflightRequests, request));
}

}  // namespace ucxx
