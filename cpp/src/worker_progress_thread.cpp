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
  std::shared_ptr<bool> stop,
  std::function<void(void)> setThreadId,
  ProgressThreadStartCallback startCallback,
  ProgressThreadStartCallbackArg startCallbackArg,
  std::shared_ptr<DelayedSubmissionCollection> delayedSubmissionCollection)
{
  /**
   * Ensure the progress thread's ID is available allowing generic callbacks to run
   * successfully even after `_progressThread == nullptr`, which may occur before
   * `WorkerProgressThreads`'s destructor completes.
   */
  setThreadId();

  if (startCallback) startCallback(startCallbackArg);

  while (!*stop) {
    delayedSubmissionCollection->processPre();

    progressFunction();

    delayedSubmissionCollection->processPost();
  }
}

WorkerProgressThread::WorkerProgressThread(
  const bool pollingMode,
  std::function<bool(void)> progressFunction,
  std::function<void(void)> signalWorkerFunction,
  std::function<void(void)> setThreadId,
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
                        _stop,
                        setThreadId,
                        _startCallback,
                        _startCallbackArg,
                        _delayedSubmissionCollection);
}

WorkerProgressThread::~WorkerProgressThread() { stop(); }

void WorkerProgressThread::stop()
{
  if (!_thread.joinable()) {
    ucxx_debug("Worker progress thread not running or already stopped");
    return;
  }

  utils::CallbackNotifier callbackNotifierPre{};
  auto idPre = _delayedSubmissionCollection->registerGenericPre(
    [&callbackNotifierPre]() { callbackNotifierPre.set(); });
  _signalWorkerFunction();
  if (!callbackNotifierPre.wait(3000000000)) {
    _delayedSubmissionCollection->cancelGenericPre(idPre);
  }

  utils::CallbackNotifier callbackNotifierPost{};
  auto idPost = _delayedSubmissionCollection->registerGenericPost([this, &callbackNotifierPost]() {
    *_stop = true;
    callbackNotifierPost.set();
  });
  _signalWorkerFunction();
  if (!callbackNotifierPost.wait(3000000000)) {
    _delayedSubmissionCollection->cancelGenericPre(idPost);
  }

  _thread.join();
}

bool WorkerProgressThread::pollingMode() const { return _pollingMode; }

std::thread::id WorkerProgressThread::getId() const { return _thread.get_id(); }

bool WorkerProgressThread::isRunning() const { return _thread.get_id() != std::thread::id(); }

}  // namespace ucxx
