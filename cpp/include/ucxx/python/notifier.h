/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#if UCXX_ENABLE_PYTHON
#include <condition_variable>
#include <memory>
#include <mutex>

#include <ucxx/log.h>

namespace ucxx {

namespace python {

enum class RequestNotifierThreadState { NotRunning = 0, Running, Stopping };

class Future;

class Notifier {
 private:
  std::mutex _notifierThreadMutex{};
  std::vector<std::pair<std::shared_ptr<Future>, ucs_status_t>> _notifierThreadFutureStatus{};
  bool _notifierThreadFutureStatusReady{false};
  RequestNotifierThreadState _notifierThreadFutureStatusFinished{
    RequestNotifierThreadState::NotRunning};
  std::condition_variable _notifierThreadConditionVariable{};

  Notifier() = default;

 public:
  Notifier(const Notifier&) = delete;
  Notifier& operator=(Notifier const&) = delete;
  Notifier(Notifier&& o)               = delete;
  Notifier& operator=(Notifier&& o) = delete;

  friend std::shared_ptr<Notifier> createNotifier()
  {
    return std::shared_ptr<Notifier>(new Notifier());
  }

  void scheduleFutureNotify(std::shared_ptr<Future> future, ucs_status_t status);

  bool waitRequestNotifier();

  void runRequestNotifier();

  void stopRequestNotifierThread();
};

}  // namespace python

}  // namespace ucxx
#endif
