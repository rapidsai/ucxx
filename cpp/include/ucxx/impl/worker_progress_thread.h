/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucxx/log.h>
#include <ucxx/worker_progress_thread.h>

namespace ucxx {

void WorkerProgressThread::progressUntilSync(
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

WorkerProgressThread::WorkerProgressThread(
  const bool pollingMode,
  std::function<bool(void)> progressFunction,
  ProgressThreadStartCallback startCallback,
  ProgressThreadStartCallbackArg startCallbackArg,
  std::shared_ptr<DelayedNotificationRequestCollection> delayedNotificationRequestCollection)
  : _pollingMode(pollingMode), _startCallback(startCallback), _startCallbackArg(startCallbackArg)
{
  _thread = std::thread(WorkerProgressThread::progressUntilSync,
                        progressFunction,
                        std::ref(_stop),
                        _startCallback,
                        _startCallbackArg,
                        delayedNotificationRequestCollection);
}

WorkerProgressThread::~WorkerProgressThread()
{
  if (!_thread.joinable()) {
    ucxx_warn("Worker progress thread not running or already stopped");
    return;
  }

  _stop = true;
  _thread.join();
}

bool WorkerProgressThread::pollingMode() const { return _pollingMode; }

}  // namespace ucxx
