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

typedef std::function<void(std::shared_ptr<void>)> DelayedNotificationRequestCallback;
typedef std::shared_ptr<void> DelayedNotificationRequestCallbackData;

class DelayedNotificationRequest {
 private:
  DelayedNotificationRequestCallback _callback{nullptr};
  DelayedNotificationRequestCallbackData _callbackData{nullptr};

 public:
  DelayedNotificationRequest(DelayedNotificationRequestCallback callback,
                             DelayedNotificationRequestCallbackData callbackData)
    : _callback(callback), _callbackData(callbackData)
  {
  }

  std::pair<DelayedNotificationRequestCallback, DelayedNotificationRequestCallbackData> get()
  {
    return std::pair(_callback, _callbackData);
  }
};

typedef std::shared_ptr<DelayedNotificationRequest> DelayedNotificationRequestCallbackPtr;

class DelayedNotificationRequestCollection {
 private:
  std::vector<DelayedNotificationRequestCallbackPtr> _collection{};
  std::mutex _mutex{};

 public:
  DelayedNotificationRequestCollection() = default;

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
        auto callbackPair = dnr->get();
        auto callback     = callbackPair.first;
        auto callbackData = callbackPair.second;

        ucxx_trace_req("Submitting request: %p %p",
                       callback.target<void (*)(std::shared_ptr<void>)>(),
                       callbackData.get());

        if (callback) callback(callbackData);
      }
    }
  }

  void registerRequest(DelayedNotificationRequestCallback callback,
                       DelayedNotificationRequestCallbackData callbackData)
  {
    auto r = std::make_shared<DelayedNotificationRequest>(callback, callbackData);

    {
      std::lock_guard<std::mutex> lock(_mutex);
      _collection.push_back(r);
    }
    ucxx_trace_req("Registered submit request: %p %p",
                   callback.target<void (*)(std::shared_ptr<void>)>(),
                   callbackData.get());
  }
};

}  // namespace ucxx
