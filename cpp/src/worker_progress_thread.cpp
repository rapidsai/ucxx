/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>

#include <ucxx/log.h>
#include <ucxx/utils/callback_notifier.h>
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
    delayedSubmissionCollection->processPre();

    progressFunction();

    delayedSubmissionCollection->processPost();
  }
}

WorkerProgressThread::WorkerProgressThread(
  const bool pollingMode,
  std::function<bool(void)> progressFunction,
  std::function<void(void)> signalWorkerFunction,
  ProgressThreadStartCallback startCallback,
  ProgressThreadStartCallbackArg startCallbackArg,
  std::shared_ptr<DelayedSubmissionCollection> delayedSubmissionCollection)
  : _pollingMode(pollingMode),
    _signalWorkerFunction(signalWorkerFunction),
    _startCallback(startCallback),
    _startCallbackArg(startCallbackArg),
    _delayedSubmissionCollection(delayedSubmissionCollection)
{
  _thread = std::thread(WorkerProgressThread::progressUntilSync,
                        progressFunction,
                        std::ref(_stop),
                        _startCallback,
                        _startCallbackArg,
                        _delayedSubmissionCollection);
}

WorkerProgressThread::~WorkerProgressThread()
{
  if (!_thread.joinable()) {
    ucxx_warn("Worker progress thread not running or already stopped");
    return;
  }

  utils::CallbackNotifier notifier{};
  _delayedSubmissionCollection->registerGenericPre([&notifier]() { notifier.set(); });
  _signalWorkerFunction();
  notifier.wait();
  _delayedSubmissionCollection->registerGenericPost([this, &notifier]() {
    _stop = true;
    notifier.set();
  });
  _signalWorkerFunction();
  notifier.wait();

  _thread.join();
}

bool WorkerProgressThread::pollingMode() const { return _pollingMode; }

std::thread::id WorkerProgressThread::getId() const { return _thread.get_id(); }

}  // namespace ucxx
