/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <functional>
#include <mutex>
#include <thread>

#include <ucxx/log.h>

namespace ucxx {

typedef std::function<void(void*)> ProgressThreadStartCallback;
typedef void* ProgressThreadStartCallbackArg;

class UCXXWorkerProgressThread {
 private:
  std::thread _thread{};
  bool _stop{false};
  bool _pollingMode{false};
  ProgressThreadStartCallback _startCallback{nullptr};
  ProgressThreadStartCallbackArg _startCallbackArg{nullptr};

  static void progressUntilSync(
    std::function<bool(void)> progressFunction,
    const bool& stop,
    ProgressThreadStartCallback startCallback,
    ProgressThreadStartCallbackArg startCallbackArg,
    std::shared_ptr<DelayedNotificationRequestCollection> delayedNotificationRequestCollection)
  {
    if (startCallback) startCallback(startCallbackArg);

    while (!stop) {
      if (delayedNotificationRequestCollection != nullptr)
        delayedNotificationRequestCollection->process();

      progressFunction();
    }
  }

 public:
  UCXXWorkerProgressThread() = delete;

  UCXXWorkerProgressThread(
    const bool pollingMode,
    std::function<bool(void)> progressFunction,
    ProgressThreadStartCallback startCallback,
    ProgressThreadStartCallbackArg startCallbackArg,
    std::shared_ptr<DelayedNotificationRequestCollection> delayedNotificationRequestCollection)
    : _pollingMode(pollingMode), _startCallback(startCallback), _startCallbackArg(startCallbackArg)
  {
    _thread = std::thread(UCXXWorkerProgressThread::progressUntilSync,
                          progressFunction,
                          std::ref(_stop),
                          _startCallback,
                          _startCallbackArg,
                          delayedNotificationRequestCollection);
  }

  ~UCXXWorkerProgressThread()
  {
    if (!_thread.joinable()) {
      ucxx_warn("Worker progress thread not running or already stopped");
      return;
    }

    _stop = true;
    _thread.join();
  }

  bool pollingMode() const { return _pollingMode; }
};

}  // namespace ucxx
