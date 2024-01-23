/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include <ucxx/delayed_submission.h>

namespace ucxx {

/**
 * @brief A user-defined function used to wake the worker.
 *
 * A user-defined function signaling worker to wake the progress event.
 */
typedef std::function<void(void)> SignalWorkerFunction;

/**
 * @brief A user-defined function to execute at the start of the progress thread.
 *
 * A user-defined function to execute at the start of the progress thread, used for example
 * to ensure resources are properly set for the thread, such as a CUDA context.
 */
typedef std::function<void(void*)> ProgressThreadStartCallback;

/**
 * @brief Data for the user-defined function provided to progress thread start callback.
 *
 * Data passed to the user-defined function provided to the callback executed when the
 * progress thread starts, which the custom user-defined function may act upon.
 */
typedef void* ProgressThreadStartCallbackArg;

/**
 * @brief A thread to progress a `ucxx::Worker`.
 *
 * A thread to progress the `ucxx::Worker`, thus moving all such blocking operations out of
 * the main program thread. It may also be used to execute submissions, such as from
 * `ucxx::Request` objects, therefore also moving any blocking costs of those to this
 * thread, as well as generic pre-progress and post-progress callbacks that can be used by
 * the program to block until that stage is reached.
 */
class WorkerProgressThread {
 private:
  std::thread _thread{};     ///< Thread object
  bool _stop{false};         ///< Signal to stop on next iteration
  bool _pollingMode{false};  ///< Whether thread will use polling mode to progress
  SignalWorkerFunction _signalWorkerFunction{
    nullptr};  ///< Function signaling worker to wake the progress event (when _pollingMode is
               ///< `false`)
  ProgressThreadStartCallback _startCallback{
    nullptr};  ///< Callback to execute at start of the progress thread
  ProgressThreadStartCallbackArg _startCallbackArg{
    nullptr};  ///< Argument to pass to start callback
  std::shared_ptr<DelayedSubmissionCollection> _delayedSubmissionCollection{
    nullptr};  ///< Collection of enqueued delayed submissions

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
   * made private to ensure all UCXX objects are shared pointers for correct
   * lifetime management.
   *
   * This thread runs asynchronously with the main application thread. If you require
   * cross-thread synchronization (for example when tearing down the thread or canceling
   * requests), use the generic pre and post callbacks with a `CallbackNotifier` that
   * synchronizes with the application thread. Since the worker progress itself may change
   * state, it is usually the case that synchronization is needed in both pre and post
   * callbacks.
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
   * @param[in] signalWorkerFunction        user-defined function to wake the worker
   *                                        progress event (when `pollingMode` is `false`).
   * @param[in] startCallback               user-defined callback function to be executed
   *                                        at the start of the progress thread.
   * @param[in] startCallbackArg            an argument to be passed to the start callback.
   * @param[in] delayedSubmissionCollection collection of delayed submissions to be
   *                                        processed during progress.
   */
  WorkerProgressThread(const bool pollingMode,
                       std::function<bool(void)> progressFunction,
                       std::function<void(void)> signalWorkerFunction,
                       ProgressThreadStartCallback startCallback,
                       ProgressThreadStartCallbackArg startCallbackArg,
                       std::shared_ptr<DelayedSubmissionCollection> delayedSubmissionCollection);

  /**
   * @brief `ucxx::WorkerProgressThread` destructor.
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

  /**
   * @brief Returns the ID of the progress thread.
   *
   * @returns the progress thread ID.
   */
  std::thread::id getId() const;
};

}  // namespace ucxx
