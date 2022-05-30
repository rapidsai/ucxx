/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#if UCXX_ENABLE_PYTHON
#include <memory>
#include <mutex>

#include <ucxx/log.h>

namespace ucxx {

enum RequestNotifierThreadState {
  RequestNotifierThreadNotRunning,
  RequestNotifierThreadRunning,
  RequestNotifierThreadStopping
};

class PythonFuture;

class Notifier {
 private:
  std::mutex _notifierThreadMutex{};
  std::vector<std::pair<std::shared_ptr<PythonFuture>, ucs_status_t>> _notifierThreadFutureStatus{};
  bool _notifierThreadFutureStatusReady{false};
  RequestNotifierThreadState _notifierThreadFutureStatusFinished{RequestNotifierThreadNotRunning};
  std::condition_variable _notifierThreadConditionVariable{};

  Notifier() = default;

 public:
  Notifier(const Notifier&) = delete;
  Notifier& operator=(Notifier const&) = delete;
  Notifier(Notifier&& o)               = delete;
  Notifier& operator=(Notifier&& o) = delete;

  template <class... Args>
  friend std::shared_ptr<Notifier> createNotifier(Args&&... args)
  {
    return std::shared_ptr<Notifier>(new Notifier(std::forward<Args>(args)...));
  }

  void schedulePythonFutureNotifyEmpty()
  {
    ucxx_trace_req("Notifer::schedulePythonFutureNotifyEmpty(): %p", this);
  }

  void schedulePythonFutureNotify(std::shared_ptr<PythonFuture> future, ucs_status_t status);

  bool waitRequestNotifier()
  {
    ucxx_trace_req("Notifier::waitRequestNotifier()");

    if (_notifierThreadFutureStatusFinished == RequestNotifierThreadStopping) {
      _notifierThreadFutureStatusFinished = RequestNotifierThreadNotRunning;
      return true;
    }

    std::unique_lock lock(_notifierThreadMutex);
    _notifierThreadConditionVariable.wait(lock, [this] {
      return _notifierThreadFutureStatusReady ||
             _notifierThreadFutureStatusFinished == RequestNotifierThreadStopping;
    });

    ucxx_trace_req("Notifier::waitRequestNotifier() unlock: %d %d",
                   _notifierThreadFutureStatusReady,
                   _notifierThreadFutureStatusFinished);
    _notifierThreadFutureStatusReady = false;

    return false;
  }

  void runRequestNotifier();

  void stopRequestNotifierThread()
  {
    {
      std::lock_guard lock(_notifierThreadMutex);
      _notifierThreadFutureStatusFinished = RequestNotifierThreadStopping;
    }
    _notifierThreadConditionVariable.notify_all();
  }
};

}  // namespace ucxx
#endif
