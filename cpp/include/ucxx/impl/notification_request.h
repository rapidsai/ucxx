/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>
#include <mutex>
#include <utility>

#include <ucxx/log.h>
#include <ucxx/notification_request.h>

namespace ucxx {

NotificationRequest::NotificationRequest(const bool send,
                                         void* buffer,
                                         const size_t length,
                                         const ucp_tag_t tag)
  : _send(send), _buffer(buffer), _length(length), _tag(tag)
{
}

NotificationRequestCallback::NotificationRequestCallback(NotificationRequestCallbackType callback)
  : _callback(callback)
{
}

NotificationRequestCallbackType NotificationRequestCallback::get() { return _callback; }

void DelayedNotificationRequestCollection::process()
{
  if (_collection.size() > 0) {
    ucxx_trace_req("Submitting %lu requests", _collection.size());

    // Move _collection to a local copy in order to to hold the lock for as
    // short as possible
    decltype(_collection) toProcess;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      std::swap(_collection, toProcess);
    }

    for (auto& dnr : toProcess) {
      auto callback = dnr->get();

      ucxx_trace_req("Submitting request: %p", callback.target<void (*)(std::shared_ptr<void>)>());

      if (callback) callback();
    }
  }
}

void DelayedNotificationRequestCollection::registerRequest(NotificationRequestCallbackType callback)
{
  auto r = std::make_shared<NotificationRequestCallback>(callback);

  {
    std::lock_guard<std::mutex> lock(_mutex);
    _collection.push_back(r);
  }
  ucxx_trace_req("Registered submit request: %p",
                 callback.target<void (*)(std::shared_ptr<void>)>());
}

}  // namespace ucxx
