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
  std::thread _thread{};     ///< Thread object
  bool _stop{false};         ///< Signal to stop on next iteration
  bool _pollingMode{false};  ///< Whether thread will use polling mode to progress
  ProgressThreadStartCallback _startCallback{
    nullptr};  ///< Callback to execute at start of the progress thread
  ProgressThreadStartCallbackArg _startCallbackArg{
    nullptr};  ///< Argument to pass to start callback

  /**
   * @brief The function executed in the new thread.
   *
   * This function ensures the `startCallback` is executed once at the start of the thread,
   * subsequently starting a continuous loop that processes any delayed submission requests
   * that are pending in the `delayedSubmissionCollection` followed by the execution of the
   * `progressFunction`, the loop repeats until `stop` is set.
   *
   * @param[in] progressFunction            user-defined progress function implementation.
   * @param[in] stop                        reference to the stop signal causing the
   *                                        progress loop to terminate.
   * @param[in] startCallback               user-defined callback function to be executed
   *                                        at the start of the progress thread.
   * @param[in] startCallbackArg            an argument to be passed to the start callback.
   * @param[in] delayedSubmissionCollection collection of delayed submissions to be
   *                                        processed during progress.
   */
  static void progressUntilSync(
    std::function<bool(void)> progressFunction,
    const bool& stop,
    ProgressThreadStartCallback startCallback,
    ProgressThreadStartCallbackArg startCallbackArg,
    std::shared_ptr<DelayedSubmissionCollection> delayedSubmissionCollection);

 public:
  WorkerProgressThread() = delete;

  /**
   * @brief Constructor of `shared_ptr<ucxx::Worker>`.
   *
   * The constructor for a `shared_ptr<ucxx::Worker>` object. The default constructor is
   * made private to ensure all UCXX objects are shared pointers and correct
   * lifetime management.
   *
   * @code{.cpp}
   * // context is `std::shared_ptr<ucxx::Context>`
   * auto worker = context->createWorker(false);
   *
   * // Equivalent to line above
   * // auto worker = ucxx::createWorker(context, false);
   * @endcode
   *
   * @param[in] pollingMode                 whether the thread should use polling mode to
   *                                        progress.
   * @param[in] progressFunction            user-defined progress function implementation.
   * @param[in] startCallback               user-defined callback function to be executed
   *                                        at the start of the progress thread.
   * @param[in] startCallbackArg            an argument to be passed to the start callback.
   * @param[in] delayedSubmissionCollection collection of delayed submissions to be
   *                                        processed during progress.
   */
  WorkerProgressThread(const bool pollingMode,
                       std::function<bool(void)> progressFunction,
                       ProgressThreadStartCallback startCallback,
                       ProgressThreadStartCallbackArg startCallbackArg,
                       std::shared_ptr<DelayedSubmissionCollection> delayedSubmissionCollection);

  /**
   * @brief `ucxx::WorkerProgressThread destructor.
   *
   * Raises the stop signal and joins the thread.
   */
  ~WorkerProgressThread();

  /**
   * @brief Returns whether the thread was created for polling progress mode.
   *
   * @returns Whether polling mode is enabled.
   */
  bool pollingMode() const;
};

}  // namespace ucxx
