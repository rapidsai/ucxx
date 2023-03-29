/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>

#include <ucxx/log.h>
#include <ucxx/worker_progress_thread.h>

namespace ucxx {

void WorkerProgressThread::progressUntilSync(
  std::function<bool(void)> progressFunction,
  const bool& stop,
  ProgressThreadStartCallback startCallback,
  ProgressThreadStartCallbackArg startCallbackArg,
  std::shared_ptr<DelayedSubmissionCollection> delayedSubmissionCollection)
{
  if (startCallback) startCallback(startCallbackArg);

  while (!stop) {
    if (delayedSubmissionCollection != nullptr) delayedSubmissionCollection->process();

    progressFunction();
  }
}

WorkerProgressThread::WorkerProgressThread(
  const bool pollingMode,
  std::function<bool(void)> progressFunction,
  ProgressThreadStartCallback startCallback,
  ProgressThreadStartCallbackArg startCallbackArg,
  std::shared_ptr<DelayedSubmissionCollection> delayedSubmissionCollection)
  : _pollingMode(pollingMode), _startCallback(startCallback), _startCallbackArg(startCallbackArg)
{
  _thread = std::thread(WorkerProgressThread::progressUntilSync,
                        progressFunction,
                        std::ref(_stop),
                        _startCallback,
                        _startCallbackArg,
                        delayedSubmissionCollection);
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
