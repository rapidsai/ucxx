/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
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

  NotificationRequest(const bool send, void* buffer, const size_t length, const ucp_tag_t tag = 0);
};

class NotificationRequestCallback {
 private:
  NotificationRequestCallbackType _callback{nullptr};

 public:
  NotificationRequestCallback(NotificationRequestCallbackType callback);

  NotificationRequestCallbackType get();
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

  void process();

  void registerRequest(NotificationRequestCallbackType callback);
};

}  // namespace ucxx
