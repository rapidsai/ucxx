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
  return std::shared_ptr<UCXXRequestStream>(
    new UCXXRequestStream(worker, endpoint, send, buffer, length));
}

}  // namespace ucxx
