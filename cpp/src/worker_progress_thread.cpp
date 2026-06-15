/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>

#include <ucxx/log.h>
#include <ucxx/utils/callback_notifier.h>
#include <ucxx/worker_progress_thread.h>

namespace ucxx {
namespace {

constexpr uint64_t stopTimeoutNs{3000000000};
constexpr uint64_t stopSignalIntervalNs{100000000};
constexpr std::chrono::nanoseconds stopTimeout{stopTimeoutNs};
constexpr std::chrono::nanoseconds stopSignalInterval{stopSignalIntervalNs};

}  // namespace

void WorkerProgressThread::progressUntilSync(
  std::function<bool(void)> progressFunction,
  std::shared_ptr<std::atomic_bool> stop,
  std::shared_ptr<std::atomic_bool> finished,
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

  while (!stop->load(std::memory_order_acquire)) {
    delayedSubmissionCollection->processPre();

    progressFunction();

    delayedSubmissionCollection->processPost();
  }

  finished->store(true, std::memory_order_release);
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
                        _finished,
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

  auto directStop = [this]() {
    _stop->store(true, std::memory_order_release);
    _signalWorkerFunction();
  };

  auto waitForThreadFinished = [this]() {
    const auto deadline = std::chrono::steady_clock::now() + stopTimeout;
    while (!_finished->load(std::memory_order_acquire)) {
      _signalWorkerFunction();

      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) return false;

      auto sleepPeriod = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now);
      if (sleepPeriod > stopSignalInterval) sleepPeriod = stopSignalInterval;
      std::this_thread::sleep_for(sleepPeriod);
    }

    return true;
  };

  utils::CallbackNotifier callbackNotifierPre{};
  auto idPre = _delayedSubmissionCollection->registerGenericPre(
    [&callbackNotifierPre]() { callbackNotifierPre.set(); });
  _signalWorkerFunction();
  if (!callbackNotifierPre.wait(stopTimeoutNs, _signalWorkerFunction, stopSignalIntervalNs)) {
    try {
      _delayedSubmissionCollection->cancelGenericPre(idPre);
    } catch (const std::runtime_error& e) {
      ucxx_warn("Could not cancel progress thread stop pre callback: %s", e.what());
    }
  }

  bool directStopRequired = false;
  utils::CallbackNotifier callbackNotifierPost{};
  auto idPost = _delayedSubmissionCollection->registerGenericPost([this, &callbackNotifierPost]() {
    _stop->store(true, std::memory_order_release);
    callbackNotifierPost.set();
  });
  _signalWorkerFunction();
  if (!callbackNotifierPost.wait(stopTimeoutNs, _signalWorkerFunction, stopSignalIntervalNs)) {
    directStopRequired = true;
    try {
      _delayedSubmissionCollection->cancelGenericPost(idPost);
    } catch (const std::runtime_error& e) {
      ucxx_warn("Could not cancel progress thread stop post callback: %s", e.what());
    }
  }

  if (directStopRequired) {
    ucxx_warn(
      "Timed out waiting for worker progress thread stop callback; requesting direct stop");
    directStop();
  }

  if (!waitForThreadFinished()) {
    ucxx_error(
      "Worker progress thread did not stop within %lu ns; terminating to avoid an indefinite join",
      stopTimeoutNs);
    std::terminate();
  }

  _thread.join();
}

bool WorkerProgressThread::pollingMode() const { return _pollingMode; }

std::thread::id WorkerProgressThread::getId() const { return _thread.get_id(); }

bool WorkerProgressThread::isRunning() const { return _thread.get_id() != std::thread::id(); }

}  // namespace ucxx
