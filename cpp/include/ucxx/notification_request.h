/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <ucxx/log.h>

namespace ucxx {

class NotificationRequest;

typedef std::function<void()> NotificationRequestCallbackType;

class NotificationRequest {
 public:
  bool _send{false};
  void* _buffer{nullptr};
  size_t _length{0};
  ucp_tag_t _tag{0};

  NotificationRequest() = delete;

  NotificationRequest(const bool send, void* buffer, const size_t length, const ucp_tag_t tag = 0)
    : _send(send), _buffer(buffer), _length(length), _tag(tag)
  {
  }
};

class NotificationRequestCallback {
 private:
  NotificationRequestCallbackType _callback{nullptr};

 public:
  NotificationRequestCallback(NotificationRequestCallbackType callback) : _callback(callback) {}

  NotificationRequestCallbackType get() { return _callback; }
};

typedef std::shared_ptr<NotificationRequestCallback> NotificationRequestCallbackPtrType;

class DelayedNotificationRequestCollection {
 private:
  std::vector<NotificationRequestCallbackPtrType> _collection{};
  std::mutex _mutex{};

 public:
  DelayedNotificationRequestCollection()                                            = default;
  DelayedNotificationRequestCollection(const DelayedNotificationRequestCollection&) = delete;
  DelayedNotificationRequestCollection& operator=(DelayedNotificationRequestCollection const&) =
    delete;
  DelayedNotificationRequestCollection(DelayedNotificationRequestCollection&& o) = delete;
  DelayedNotificationRequestCollection& operator=(DelayedNotificationRequestCollection&& o) =
    delete;

  void process()
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

        ucxx_trace_req("Submitting request: %p",
                       callback.target<void (*)(std::shared_ptr<void>)>());

        if (callback) callback();
      }
    }
  }

  void registerRequest(NotificationRequestCallbackType callback)
  {
    auto r = std::make_shared<NotificationRequestCallback>(callback);

    {
      std::lock_guard<std::mutex> lock(_mutex);
      _collection.push_back(r);
    }
    ucxx_trace_req("Registered submit request: %p",
                   callback.target<void (*)(std::shared_ptr<void>)>());
  }
};

}  // namespace ucxx
