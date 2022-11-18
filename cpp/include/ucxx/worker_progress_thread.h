/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <functional>
#include <mutex>
#include <thread>

#include <ucxx/delayed_submission.h>

namespace ucxx {

typedef std::function<void(void*)> ProgressThreadStartCallback;
typedef void* ProgressThreadStartCallbackArg;

class WorkerProgressThread {
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
    std::shared_ptr<DelayedSubmissionCollection> delayeSubmissionCollection);

 public:
  WorkerProgressThread() = delete;

  WorkerProgressThread(const bool pollingMode,
                       std::function<bool(void)> progressFunction,
                       ProgressThreadStartCallback startCallback,
                       ProgressThreadStartCallbackArg startCallbackArg,
                       std::shared_ptr<DelayedSubmissionCollection> delayedSubmissionCollection);

  ~WorkerProgressThread();

  bool pollingMode() const;
};

}  // namespace ucxx
