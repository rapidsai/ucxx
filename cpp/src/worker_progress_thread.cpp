/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <condition_variable>
#include <memory>
#include <mutex>

#include <ucxx/log.h>
#include <ucxx/utils/condition.h>
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

  auto statusMutex             = std::make_shared<std::mutex>();
  auto statusConditionVariable = std::make_shared<std::condition_variable>();
  auto pre                     = std::make_shared<bool>(false);
  auto post                    = std::make_shared<bool>(false);

  auto setterPre = [pre]() { *pre = true; };
  auto getterPre = [pre]() { return *pre; };

  _delayedSubmissionCollection->registerGenericPre(
    [&statusMutex, &statusConditionVariable, &setterPre]() {
      ucxx::utils::conditionSetter(statusMutex, statusConditionVariable, setterPre);
    });
  _signalWorkerFunction();
  ucxx::utils::conditionGetter(statusMutex, statusConditionVariable, pre, getterPre);

  auto setterPost = [this, post]() {
    _stop = true;
    *post = true;
  };
  auto getterPost = [post]() { return *post; };

  _delayedSubmissionCollection->registerGenericPost(
    [&statusMutex, &statusConditionVariable, &setterPost]() {
      ucxx::utils::conditionSetter(statusMutex, statusConditionVariable, setterPost);
    });
  _signalWorkerFunction();
  ucxx::utils::conditionGetter(statusMutex, statusConditionVariable, post, getterPost);

  _thread.join();
}

bool WorkerProgressThread::pollingMode() const { return _pollingMode; }

std::thread::id WorkerProgressThread::getId() const { return _thread.get_id(); }

}  // namespace ucxx
