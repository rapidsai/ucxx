/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/constructors.h>
#include <ucxx/context.h>
#include <ucxx/delayed_submission.h>
#include <ucxx/future.h>
#include <ucxx/inflight_requests.h>
#include <ucxx/notifier.h>
#include <ucxx/typedefs.h>
#include <ucxx/worker_progress_thread.h>

namespace ucxx {

class Address;
class Buffer;
class Endpoint;
class Listener;
class RequestAm;

namespace internal {
class AmData;
}  // namespace internal

/**
 * @brief Component encapsulating a UCP worker.
 *
 * The UCP layer provides a handle to access workers in form of `ucp_worker_h` object,
 * this class encapsulates that object and provides methods to simplify its handling.
 */
class Worker : public Component {
 private:
  ucp_worker_h _handle{nullptr};        ///< The UCP worker handle
  int _epollFileDescriptor{-1};         ///< The epoll file descriptor
  int _workerFileDescriptor{-1};        ///< The worker file descriptor
  std::mutex _inflightRequestsMutex{};  ///< Mutex to access the inflight requests pool
  std::unique_ptr<InflightRequests> _inflightRequests{
    std::make_unique<InflightRequests>()};  ///< The inflight requests
  std::mutex
    _inflightRequestsToCancelMutex{};  ///< Mutex to access the inflight requests to cancel pool
  std::unique_ptr<InflightRequests> _inflightRequestsToCancel{
    std::make_unique<InflightRequests>()};  ///< The inflight requests scheduled to be canceled
  WorkerProgressThread _progressThread{};   ///< The progress thread object
  std::thread::id _progressThreadId{};      ///< The progress thread ID
  std::function<void(void*)> _progressThreadStartCallback{
    nullptr};  ///< The callback function to execute at progress thread start
  void* _progressThreadStartCallbackArg{
    nullptr};  ///< The argument to be passed to the progress thread start callback
  std::shared_ptr<DelayedSubmissionCollection> _delayedSubmissionCollection{
    nullptr};  ///< Collection of enqueued delayed submissions

  friend std::shared_ptr<RequestAm> createRequestAm(
    std::shared_ptr<Endpoint> endpoint,
    const std::variant<data::AmSend, data::AmReceive> requestData,
    const bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);

 protected:
  bool _enableFuture{
    false};  ///< Boolean identifying whether the worker was created with future capability
  std::mutex _futuresPoolMutex{};  ///< Mutex to access the futures pool
  std::queue<std::shared_ptr<Future>>
    _futuresPool{};  ///< Futures pool to prevent running out of fresh futures
  std::shared_ptr<Notifier> _notifier{nullptr};  ///< Notifier object
  std::shared_ptr<internal::AmData>
    _amData;  ///< Worker data made available to Active Messages callback

 private:
  /**
   * @brief Drain the worker for uncaught tag messages received.
   *
   * Called by the destructor, any uncaught tag messages received will be drained so as
   * not to generate UCX warnings.
   */
  void drainWorkerTagRecv();

  /**
   * @brief Get active message receive request.
   *
   * Returns an active message request from the pool if the worker has already begun
   * handling a request with the active messages callback, otherwise creates a new request
   * that is later populated with status and buffer by the active messages callback.
   *
   * @param[in] ep  the endpoint handle where receiving the message, the same handle that
   *                will later be used to reply to the message.
   * @param[in] createAmRecvRequestFunction function to create a new request if one is not
   *                                        already available in the pool.
   *
   * @returns Request to be subsequently checked for the completion state and data.
   */
  [[nodiscard]] std::shared_ptr<RequestAm> getAmRecv(
    ucp_ep_h ep, std::function<std::shared_ptr<RequestAm>()> createAmRecvRequestFunction);

  /**
   * @brief Stop the progress thread if running without raising warnings.
   *
   * Called by the destructor, will stop the progress thread if running without
   * raising warnings.
   */
  void stopProgressThreadNoWarn();

  /**
   * @brief Register an inflight request.
   *
   * Called each time a new transfer request is made by the `Worker`, such that it may
   * be canceled when necessary.
   *
   * @param[in] request the request to register.
   *
   * @return the request that was registered (i.e., the `request` argument itself).
   */
  [[nodiscard]] std::shared_ptr<Request> registerInflightRequest(std::shared_ptr<Request> request);

  /**
   * @brief Progress the worker until all communication events are completed.
   *
   * Iteratively calls `progressOnce()` until all communication events are completed.
   *
   * @returns whether any communication events have been progressed.
   */
  bool progressPending();

 protected:
  /**
   * @brief Protected constructor of `ucxx::Worker`.
   *
   * This is the internal implementation of `ucxx::Worker` constructor, made protected not
   * to be called directly. Instead the user should call `context::createWorker()` or
   * `ucxx::createWorker()` (or `ucxx::createPythonWorker` for the Python-enabled
   * implementation).
   *
   *
   * @param[in] context the context from which to create the worker.
   * @param[in] enableDelayedSubmission if `true`, each `ucxx::Request` will not be
   *                                    submitted immediately, but instead delayed to
   *                                    the progress thread. Requires use of the
   *                                    progress thread.
   * @param[in] enableFuture if `true`, notifies the future associated with each
   *                         `ucxx::Request`, currently used only by `ucxx::python::Worker`.
   */
  explicit Worker(std::shared_ptr<Context> context,
                  const bool enableDelayedSubmission = false,
                  const bool enableFuture            = false);

 public:
  Worker()                         = delete;
  Worker(const Worker&)            = delete;
  Worker& operator=(Worker const&) = delete;
  Worker(Worker&& o)               = delete;
  Worker& operator=(Worker&& o)    = delete;

  /**
   * @brief Constructor of `shared_ptr<ucxx::Worker>`.
   *
   * The constructor for a `shared_ptr<ucxx::Worker>` object. The default constructor is
   * made private to ensure all UCXX objects are shared pointers for correct
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
   * @param[in] context the context from which to create the worker.
   * @param[in] enableDelayedSubmission if `true`, each `ucxx::Request` will not be
   *                                    submitted immediately, but instead delayed to
   *                                    the progress thread. Requires use of the
   *                                    progress thread.
   * @param[in] enableFuture if `true`, notifies the future associated with each
   *                         `ucxx::Request`, currently used only by `ucxx::python::Worker`.
   * @returns The `shared_ptr<ucxx::Worker>` object
   */
  friend std::shared_ptr<Worker> createWorker(std::shared_ptr<Context> context,
                                              const bool enableDelayedSubmission,
                                              const bool enableFuture);

  /**
   * @brief `ucxx::Worker` destructor.
   */
  virtual ~Worker();

  /**
   * @brief Get the underlying `ucp_worker_h` handle.
   *
   * Lifetime of the `ucp_worker_h` handle is managed by the `ucxx::Worker` object and
   * its ownership is non-transferrable. Once the `ucxx::Worker` is destroyed the handle
   * is not valid anymore, it is the user's responsibility to ensure the owner's lifetime
   * while using the handle.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   * ucp_worker_h workerHandle = worker->getHandle();
   * @endcode
   *
   * @returns The underlying `ucp_worker_h` handle.
   */
  [[nodiscard]] ucp_worker_h getHandle();

  /**
   * @brief Get information about the underlying `ucp_worker_h` object.
   *
   * Convenience wrapper for `ucp_worker_print_info()` to get information about the
   * underlying UCP worker handle and return it as a string.
   *
   * @returns String containing information about the UCP worker.
   */
  [[nodiscard]] std::string getInfo();

  /**
   * @brief Initialize blocking progress mode.
   *
   * Initialize blocking progress mode, creates internal file descriptors to handle blocking
   * progress by waiting for the UCP worker to notify the file descriptors. This method is
   * supposed to be called when usage of `progressWorkerEvent()` is intended, before the
   * first call to `progressWorkerEvent()`. If using polling mode only via
   * `progress()`/`progressOnce()` calls or wake-up with `waitProgress()`, this method should
   * not be called.
   *
   * In blocking mode, the user should call `progressWorkerEvent()` to block and then progress
   * the worker as new events arrive. `wakeProgressEvent()` may be called to forcefully wake
   * this method, for example to shutdown the application.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   *
   * // Block until UCX's wakes up for an incoming event, then fully progresses the
   * // worker
   * worker->initBlockingProgressMode();
   * worker->progressWorkerEvent();
   *
   * // All events have been progressed.
   * @endcode
   *
   * @throws std::ios_base::failure if creating any of the file descriptors or setting their
   *                                statuses.
   */
  void initBlockingProgressMode();

  /**
   * @brief Get the epoll file descriptor associated with the worker.
   *
   * Get the epoll file descriptor associated with the worker when running in blocking mode.
   * The worker only has an associated epoll file descriptor after
   * `initBlockingProgressMode()` is executed.
   *
   * The file descriptor is destroyed as part of the `ucxx::Worker` destructor, thus any
   * reference to it shall not be used after that.
   *
   * @throws std::runtime_error if `initBlockingProgressMode()` was not executed to run the
   *                            worker in blocking progress mode.
   *
   * @returns the file descriptor.
   */
  int getEpollFileDescriptor();

  /**
   * @brief Arm the UCP worker.
   *
   * Wrapper for `ucp_worker_arm`, checking its return status for errors and raising an
   * exception if an error occurred.
   *
   * @throws ucxx::Error if an error occurred while attempting to arm the worker.
   *
   * @returns `true` if worker was armed successfully, `false` if its status was `UCS_ERR_BUSY`.
   */
  bool arm();

  /**
   * @brief Progress worker event while in blocking progress mode.
   *
   * Blocks until a new worker event has happened and the worker notifies the file descriptor
   * associated with it, or `epollTimeout` has elapsed. Requires blocking progress mode to
   * be initialized with `initBlockingProgressMode()` before the first call to this method.
   * Additionally ensure inflight messages pending for cancelation are canceled.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   *
   * // Block until UCX's wakes up for an incoming event, then fully progresses the
   * // worker
   * worker->initBlockingProgressMode();
   * worker->progressWorkerEvent();
   *
   * // All events have been progressed.
   * @endcode
   *
   * @param[in] epollTimeout  timeout in ms when waiting for worker event, or -1 to block
   *                          indefinitely.
   *
   * @throws std::ios_base::failure if creating any of the file descriptors or setting their
   *                                statuses.
   *
   * @returns `true` if any communication was progressed, `false` otherwise.
   */
  bool progressWorkerEvent(const int epollTimeout = -1);

  /**
   * @brief Signal the worker that an event happened.
   *
   * Signals that an event has happened while, causing both either `progressWorkerEvent()`
   * or `waitProgress()` to immediately wake-up.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   *
   * void progressThread() {
   *   // Block until UCX's wakes up for an incoming event, then fully progresses the
   *   // worker.
   *   worker->initBlockingProgressMode();
   *   worker->progressWorkerEvent();
   *
   *   // Will reach this point and exit after 3 seconds
   * }
   *
   * void otherThread() {
   *   // Signals the worker after 3 seconds
   *   std::this_thread::sleep_for(std::chrono::seconds(3));
   *   worker->signal();
   * }
   *
   * void mainThread() {
   *   t1 = std::thread(progressThread);
   *   t2 = std::thread(otherThread);
   *   t1.join();
   *   t2.join();
   * }
   * @endcode
   *
   * @throws ucxx::Error if an error occurred while attempting to signal the worker.
   */
  void signal();

  /**
   * @brief Block until an event has happened, then progresses.
   *
   * Blocks until an event has happened as part of UCX's wake-up mechanism and progress
   * the worker. Additionally ensure inflight messages pending for cancelation are canceled.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   *
   * // Block until UCX's wakes up for an incoming event, then fully progresses the
   * // worker
   * worker->waitProgress();
   * worker->progress();
   *
   * // All events have been progressed.
   * @endcode
   *
   * @throws ucxx::Error if an error occurred while attempting to arm the worker.
   *
   * @returns `true` if any communication was progressed, `false` otherwise.
   */
  bool waitProgress();

  /**
   * @brief Progress the worker only once.
   *
   * Wrapper for `ucp_worker_progress`.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   * while (!worker->progressOnce()) ;
   *
   * // All events have been progressed.
   * @endcode
   *
   * @returns `true` if any communication was progressed, `false` otherwise.
   */
  bool progressOnce();

  /**
   * @brief Progress the worker until all communication events are completed.
   *
   * Iteratively calls `progressOnce()` until all communication events are completed.
   * Additionally ensure inflight messages pending for cancelation are canceled.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   * worker->progress();
   *
   * // All events have been progressed and inflight pending for cancelation were canceled.
   * @endcode
   *
   * @returns whether any communication events have been progressed.
   */
  bool progress();

  /**
   * @brief Register delayed request submission.
   *
   * Register `ucxx::Request` for delayed submission. When the `ucxx::Worker` is created
   * with `enableDelayedSubmission=true`, calling actual UCX transfer routines will not
   * happen immediately and instead will be submitted later by the worker thread.
   *
   * The purpose of this method is to offload as much as possible any work to the worker
   * thread, thus decreasing computation on the caller thread, but potentially increasing
   * transfer latency.
   *
   * @param[in] request  the request to which the callback belongs, ensuring it remains
   *                     alive until the callback is invoked.
   * @param[in] callback the callback set to execute the UCP transfer routine during the
   *                     worker thread loop.
   */
  void registerDelayedSubmission(std::shared_ptr<Request> request,
                                 DelayedSubmissionCallbackType callback);

  /**
   * @brief Register callback to be executed in progress thread before progressing.
   *
   * Register callback to be executed in the current or next iteration of the progress
   * thread before the worker is progressed. There is no guarantee that the callback will
   * be executed in the current or next iteration, this depends on where the progress thread
   * is in the current iteration when this callback is registered.
   *
   * The purpose of this method is to schedule operations to be executed in the progress
   * thread, such as endpoint creation and closing, so that progressing doesn't ever need
   * to occur in the application thread when using a progress thread.
   *
   * If `period` is `0` this is a blocking call that only returns when the callback has been
   * executed and will always return `true`, and if `period` is a positive integer the time
   * in nanoseconds will be waited for the callback to complete and return `true` in the
   * successful case or `false` otherwise. However, if the callback is not cancelable
   * anymore (i.e., it has already started), this method will keep retrying and may never
   * return if the callback never completes, it is unsafe to return as this would allow the
   * caller to destroy the callback and its resources causing undefined behavior. `period`
   * only applies if the worker progress thread is running, otherwise the callback is
   * immediately executed.
   *
   * @param[in] callback  the callback to execute before progressing the worker.
   * @param[in] period    the time in nanoseconds to wait for the callback to complete.
   *
   * @returns `true` if the callback was successfully executed or `false` if timed out.
   */
  [[nodiscard]] bool registerGenericPre(DelayedSubmissionCallbackType callback,
                                        uint64_t period = 0);

  /**
   * @brief Register callback to be executed in progress thread before progressing.
   *
   * Register callback to be executed in the current or next iteration of the progress
   * thread after the worker is progressed. There is no guarantee that the callback will
   * be executed in the current or next iteration, this depends on where the progress thread
   * is in the current iteration when this callback is registered.
   *
   * The purpose of this method is to schedule operations to be executed in the progress
   * thread, immediately after progressing the worker completes.
   *
   * If `period` is `0` this is a blocking call that only returns when the callback has been
   * executed and will always return `true`, and if `period` is a positive integer the time
   * in nanoseconds will be waited for the callback to complete and return `true` in the
   * successful case or `false` otherwise. However, if the callback is not cancelable
   * anymore (i.e., it has already started), this method will keep retrying and may never
   * return if the callback never completes, it is unsafe to return as this would allow the
   * caller to destroy the callback and its resources causing undefined behavior. `period`
   * only applies if the worker progress thread is running, otherwise the callback is
   * immediately executed.
   *
   * @param[in] callback  the callback to execute before progressing the worker.
   * @param[in] period    the time in nanoseconds to wait for the callback to complete.
   *
   * @returns `true` if the callback was successfully executed or `false` if timed out.
   */
  [[nodiscard]] bool registerGenericPost(DelayedSubmissionCallbackType callback,
                                         uint64_t period = 0);

  /**
   * @brief Inquire if worker has been created with delayed submission enabled.
   *
   * Check whether the worker has been created with delayed submission enabled.
   *
   * @returns `true` if delayed submission is enabled, `false` otherwise.
   */
  [[nodiscard]] bool isDelayedRequestSubmissionEnabled() const;

  /**
   * @brief Inquire if worker has been created with future support.
   *
   * Check whether the worker has been created with future support.
   *
   * @returns `true` if future support is enabled, `false` otherwise.
   */
  [[nodiscard]] bool isFutureEnabled() const;

  /**
   * @brief Populate the futures pool.
   *
   * To avoid taking blocking resources (such as the Python GIL) for every new future
   * required by each `ucxx::Request`, the `ucxx::Worker` maintains a pool of futures
   * that can be acquired when a new `ucxx::Request` is created. Currently the pool has
   * a maximum size of 100 objects, and will refill once it goes under 50, otherwise
   * calling this functions results in a no-op.
   *
   * @throws std::runtime_error if future support is not implemented.
   */
  virtual void populateFuturesPool();

  /**
   * @brief Clear the futures pool.
   *
   * Clear the futures pool, ensuring all references are removed and thus avoiding
   * reference cycles that prevent the `ucxx::Worker` and other resources from cleaning
   * up on time.
   *
   * @throws std::runtime_error if future support is not implemented.
   */
  virtual void clearFuturesPool();

  /**
   * @brief Get a future from the pool.
   *
   * Get a future from the pool. If the pool is empty,
   * `ucxx::Worker::populateFuturesPool()` is called and a warning is raised, since
   * that likely means the user is missing to call the aforementioned method regularly.
   *
   * @throws std::runtime_error if future support is not implemented.
   *
   * @returns The `shared_ptr<ucxx::python::Future>` object
   */
  [[nodiscard]] virtual std::shared_ptr<Future> getFuture();

  /**
   * @brief Block until a request event.
   *
   * Blocks until some communication is completed and future is ready to be notified,
   * shutdown was initiated or a timeout occurred (only if `periodNs > 0`). This method is
   * intended for use from the notifier (such as the Python thread running it), where that
   * thread will block until one of the aforementioned events occur.
   *
   * @throws std::runtime_error if future support is not implemented.
   *
   * @returns `RequestNotifierWaitState::Ready` if some communication completed,
   *          `RequestNotifierWaitStats::Timeout` if a timeout occurred, or
   *          `RequestNotifierWaitStats::Shutdown` if shutdown has initiated.
   */
  [[nodiscard]] virtual RequestNotifierWaitState waitRequestNotifier(uint64_t periodNs);

  /**
   * @brief Notify futures of each completed communication request.
   *
   * Notifies futures of each completed communication request of their new status. This
   * method is intended to be used from the Notifier (such as the Python thread running it),
   * where the thread will call `waitRequestNotifier()` and block until some communication
   * is completed, and then call this method to notify all futures. If this is notifying
   * a Python future, the thread where this method is called from must be using the same
   * Python event loop as the thread that submitted the transfer request.
   *
   * @throws std::runtime_error if future support is not implemented.
   */
  virtual void runRequestNotifier();

  /**
   * @brief Signal the notifier to terminate.
   *
   * Signals the notifier to terminate, awakening the `waitRequestNotifier()` blocking call.
   *
   * @throws std::runtime_error if future support is not implemented.
   */
  virtual void stopRequestNotifierThread();

  /**
   * @brief Set callback to be executed at the progress thread start.
   *
   * Sets a callback that will be executed at the beginning of the progress thread. This
   * can be used to initialize any resources that are required to be available on the thread
   * the worker will be progressed from, such as a CUDA context.
   *
   * @param[in] callback function to execute during progress thread start
   * @param[in] callbackArg argument to be passed to the callback function
   */
  void setProgressThreadStartCallback(std::function<void(void*)> callback, void* callbackArg);

  /**
   * @brief Start the progress thread.
   *
   * Spawns a new thread that will take care of continuously progressing the worker. The
   * thread can progress the worker in blocking mode, using `progressWorkerEvent()` only
   * when worker events happen, or in polling mode by continuously calling `progress()`
   * (incurs in high CPU utilization).
   *
   * @param[in] pollingMode   use polling mode if `true`, or blocking mode if `false`.
   * @param[in] epollTimeout  timeout in ms when waiting for worker event, or -1 to block
   *                          indefinitely, only applicable if `pollingMode==true`.
   */
  void startProgressThread(const bool pollingMode = false, const int epollTimeout = 1);

  /**
   * @brief Stop the progress thread.
   *
   * Stop the progress thread.
   *
   * May be called by the user at any time, and also called during destructor if the
   * worker thread was ever started.
   */
  void stopProgressThread();

  /**
   * @brief Inquire if worker has a progress thread running.
   *
   * Check whether the worker currently has a progress thread running.
   *
   * @returns `true` if a progress thread is running, `false` otherwise.
   */
  [[nodiscard]] bool isProgressThreadRunning();

  /**
   * @brief Get the progress thread ID.
   *
   * Get the progress thread ID, only valid if `startProgressThread()` was called.
   *
   * @returns the progress thread ID.
   */
  [[nodiscard]] std::thread::id getProgressThreadId();

  /**
   * @brief Cancel inflight requests.
   *
   * Cancel inflight requests, returning the total number of requests that were canceled.
   * This is usually executed during the progress loop.
   *
   * If the parent worker is running a progress thread, a maximum timeout may be specified
   * for which the close operation will wait. This can be particularly important for cases
   * where the progress thread might be attempting to acquire a resource (e.g., the Python
   * GIL) while the current thread owns that resource. In particular for Python, the
   * `~Worker()` will call this method for which we can't release the GIL when the garbage
   * collector runs and destroys the object.
   *
   * @param[in] period      maximum period to wait for a generic pre/post progress thread
   *                        operation will wait for.
   * @param[in] maxAttempts maximum number of attempts to close endpoint, only applicable
   *                         if worker is running a progress thread and `period > 0`.
   *
   * @returns Number of requests that were canceled.
   */
  size_t cancelInflightRequests(uint64_t period = 0, uint64_t maxAttempts = 1);

  /**
   * @brief Schedule cancelation of inflight requests.
   *
   * Schedule inflight request to be canceled when `cancelInflightRequests()` is executed
   * the next time, usually during the progress loop. This is usually called from a
   * `ucxx::Endpoint`, for example when the error callback was called, signaling that
   * inflight requests for that endpoint will not be completed successfully and should be
   * canceled.
   *
   * @param[in] trackedRequests the requests tracked by a child of this class to be
   *                            scheduled for cancelation.
   */
  void scheduleRequestCancel(TrackedRequestsPtr trackedRequests);

  /**
   * @brief Remove reference to request from internal container.
   *
   * Remove the reference to a specific request from the internal container. This should
   * be called when a request has completed and the `ucxx::Worker` does not need to keep
   * track of it anymore. The raw pointer to a `ucxx::Request` is passed here as opposed
   * to the usual `std::shared_ptr<ucxx::Request>` used elsewhere, this is because the
   * raw pointer address is used as key to the requests reference, and this is called
   * from the object's destructor.
   *
   * @param[in] request raw pointer to the request
   */
  void removeInflightRequest(const Request* const request);

  /**
   * @brief Check for uncaught tag messages.
   *
   * Checks the worker for any uncaught tag messages. An uncaught tag message is any
   * tag message that has been fully or partially received by the worker, but not matched
   * by a corresponding `ucp_tag_recv_*` call.
   *
   * @code{.cpp}
   * // `worker` is `std::shared_ptr<ucxx::Worker>`
   * assert(!worker->tagProbe(0));
   *
   * // `ep` is a remote `std::shared_ptr<ucxx::Endpoint` to the local `worker`
   * ep->tagSend(buffer, length, 0);
   *
   * assert(worker->tagProbe(0));
   * @endcode
   *
   * @returns `true` if any uncaught messages were received, `false` otherwise.
   */
  [[nodiscard]] bool tagProbe(const Tag tag);

  /**
   * @brief Enqueue a tag receive operation.
   *
   * Enqueue a tag receive operation, returning a `std::shared<ucxx::Request>` that can
   * be later awaited and checked for errors. This is a non-blocking operation, and the
   * status of the transfer must be verified from the resulting request object before the
   * data can be consumed.
   *
   * Using a future may be requested by specifying `enableFuture` if the worker
   * implementation has support for it. If a future is requested, the application must then
   * await on this future to ensure the transfer has completed.
   *
   * @param[in] buffer            a raw pointer to pre-allocated memory where resulting
   *                              data will be stored.
   * @param[in] length            the size in bytes of the tag message to be received.
   * @param[in] tag               the tag to match.
   * @param[in] tagMask           the tag mask to use.
   * @param[in] enableFuture      whether a future should be created and subsequently
   *                              notified.
   * @param[in] callbackFunction  user-defined callback function to call upon completion.
   * @param[in] callbackData      user-defined data to pass to the `callbackFunction`.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> tagRecv(
    void* buffer,
    size_t length,
    Tag tag,
    TagMask tagMask,
    const bool enableFuture                      = false,
    RequestCallbackUserFunction callbackFunction = nullptr,
    RequestCallbackUserData callbackData         = nullptr);

  /**
   * @brief Get the address of the UCX worker object.
   *
   * Gets the address of the underlying UCX worker object, which can then be passed
   * to a remote worker, allowing creating a new endpoint to the local worker via
   * `ucxx::Worker::createEndpointFromWorkerAddress()`.
   *
   * @throws ucxx::Error if an error occurred while attempting to get the worker address.
   *
   * @returns The address of the local worker.
   */
  [[nodiscard]] std::shared_ptr<Address> getAddress();

  /**
   * @brief Create endpoint to worker listening on specific IP and port.
   *
   * Creates an endpoint to a remote worker listening on a specific IP address and port.
   * The remote worker must have an active listener created with
   * `ucxx::Worker::createListener()`.
   *
   * @code{.cpp}
   * // `worker` is `std::shared_ptr<ucxx::Worker>`
   * // Create endpoint to worker listening on `10.10.10.10:12345`.
   * auto ep = worker->createEndpointFromHostname("10.10.10.10", 12345);
   * @endcode
   *
   * @throws std::invalid_argument if the IP address or hostname is invalid.
   * @throws std::bad_alloc if there was an error allocating space to handle the address.
   * @throws ucxx::Error if an error occurred while attempting to create the endpoint.
   *
   * @param[in] ipAddress string containing the IP address of the remote worker.
   * @param[in] port port number where the remote worker is listening at.
   * @param[in] endpointErrorHandling enable endpoint error handling if `true`,
   *                                  disable otherwise.
   *
   * @returns The `shared_ptr<ucxx::Endpoint>` object
   */
  [[nodiscard]] std::shared_ptr<Endpoint> createEndpointFromHostname(
    std::string ipAddress, uint16_t port, bool endpointErrorHandling = true);

  /**
   * @brief Create endpoint to worker located at UCX address.
   *
   * Creates an endpoint to a listener-independent remote worker. The worker location is
   * identified by its UCX address, wrapped by a `std::shared_ptr<ucxx::Address>` object.
   *
   * @code{.cpp}
   * // `worker` is `std::shared_ptr<ucxx::Worker>`
   * auto localAddress = worker->getAddress();
   *
   * // pass address to remote process
   * // ...
   *
   * // receive address received from remote process
   * // ...
   *
   * // `remoteAddress` is `std::shared_ptr<ucxx::Address>`
   * auto ep = worker->createEndpointFromAddress(remoteAddress);
   * @endcode
   *
   * @throws ucxx::Error if an error occurred while attempting to create the endpoint.
   *
   * @param[in] address address of the remote UCX worker.
   * @param[in] endpointErrorHandling enable endpoint error handling if `true`,
   *                                  disable otherwise.
   *
   * @returns The `shared_ptr<ucxx::Endpoint>` object
   */
  [[nodiscard]] std::shared_ptr<Endpoint> createEndpointFromWorkerAddress(
    std::shared_ptr<Address> address, bool endpointErrorHandling = true);

  /**
   * @brief Listen for remote connections on given port.
   *
   * Starts a listener on given port. The listener allows remote processes to connect to
   * the local worker via an IP and port pair. The connection is then handle via a
   * callback specified by the user.
   *
   * @throws std::bad_alloc if there was an error allocating space to handle the address.
   * @throws ucxx::Error if an error occurred while attempting to create the listener or
   *                     to acquire its address.
   *
   * @param[in] port port number where to listen at.
   * @param[in] callback to handle each incoming connection.
   * @param[in] callbackArgs pointer to argument to pass to the callback.
   *
   * @returns The `shared_ptr<ucxx::Listener>` object
   */
  [[nodiscard]] std::shared_ptr<Listener> createListener(uint16_t port,
                                                         ucp_listener_conn_callback_t callback,
                                                         void* callbackArgs);

  /**
   * @brief Register allocator for active messages.
   *
   * Register a new allocator for active messages. By default, only one allocator is defined
   * for host memory (`UCS_MEMORY_TYPE_HOST`), and is used as a fallback when an allocator
   * for the source's memory type is unavailable. In many circumstances relying exclusively
   * on the host allocator is undesirable, for example when transferring CUDA buffers the
   * destination is always going to be a host buffer and prevent the use of transports such
   * as NVLink or InfiniBand+GPUDirectRDMA. For that reason it's important that the user
   * defines those allocators that are important for the application.
   *
   * If the `memoryType` has already been registered, the previous allocator will be
   * replaced by the new one. Be careful when doing this after transfers have started, there
   * are no guarantees that inflight messages have not already been allocated with the old
   * allocator for that type.
   *
   * @code{.cpp}
   * // context is `std::shared_ptr<ucxx::Context>`
   * auto worker = context->createWorker(false);
   *
   * worker->registerAmAllocator(`UCS_MEMORY_TYPE_CUDA`, ucxx::RMMBuffer);
   * @endcode
   *
   * @param[in] memoryType  the memory type the allocator will be used for.
   * @param[in] allocator   the allocator callable that will be used to allocate new
   *                        active message buffers.
   */
  void registerAmAllocator(ucs_memory_type_t memoryType, AmAllocatorType allocator);

  /**
   * @brief Register receiver callback for active messages.
   *
   * Register a new receiver callback for active messages. By default, active messages do
   * not execute any callbacks on the receiving end unless one is specified when sending
   * the message. If the message sender specifies a callback receiver identifier then the
   * remote receiver needs to have a callback registered with the same identifier to
   * execute when the request completes. To ensure multiple applications that do not know
   * about each other can have coexisting callbacks where receiver identifiers may have
   * the same value, an owner must be specified as well, which has the form of a string and
   * should be reasonably unique to prevent accidentally calling callbacks from a separate
   * application, thus names like "A" or "UCX" are discouraged in favor of more descriptive
   * names such as "MyFastCommsProject", and the name "ucxx" is reserved.
   *
   * Because it is impossible to predict which callback would be called in such an event,
   * the registered callback cannot be changed, thus calling this method with the same
   * given owner and identifier will throw `std::runtime_error`.
   *
   *
   * @code{.cpp}
   * // `worker` is `std::shared_ptr<ucxx::Worker>`
   * auto callback = [](std::shared_ptr<ucxx::Request> req) {
   *   std::cout << "The UCXX request address is " << (void*)req.get() << std::endl;
   * };
   *
   * worker->registerAmReceiverCallback({"MyFastApp", 0}, callback};
   * @endcode
   *
   * @throws std::runtime_error if a callback with same given owner and identifier is
   *                            already registered, or if the reserved owner name "ucxx"
   *                            is specified.
   *
   * @param[in] receiverCallbackInfo    the owner name and unique identifier of the receiver
                                        callback.
   * @param[in] callback                the callback to execute when the active message is
   *                                    received.
   */
  void registerAmReceiverCallback(AmReceiverCallbackInfo info, AmReceiverCallbackType callback);

  /**
   * @brief Check for uncaught active messages.
   *
   * Checks the worker for any uncaught active messages. An uncaught active message is any
   * active message that has been fully or partially received by the worker, but not matched
   * by a corresponding `createRequestAmRecv()` call.
   *
   * @code{.cpp}
   * // `worker` is `std::shared_ptr<ucxx::Worker>`
   * // `ep` is a remote `std::shared_ptr<ucxx::Endpoint` to the local `worker`
   * assert(!worker->amProbe(ep->getHandle()));
   *
   * ep->amSend(buffer, length);
   *
   * assert(worker->amProbe(0));
   * @endcode
   *
   * @returns `true` if any uncaught messages were received, `false` otherwise.
   */
  [[nodiscard]] bool amProbe(const ucp_ep_h endpointHandle) const;

  /**
   * @brief Enqueue a flush operation.

   * Enqueue request to flush outstanding AMO (Atomic Memory Operation) and RMA (Remote
   * Memory Access) operations on the worker, returning a pointer to a request object that
   * can be later awaited and checked for errors. This is a non-blocking operation, and its
   * status must be verified from the resulting request object to confirm the flush
   * operation has completed successfully.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> flush(
    const bool enablePythonFuture                = false,
    RequestCallbackUserFunction callbackFunction = nullptr,
    RequestCallbackUserData callbackData         = nullptr);
};

}  // namespace ucxx
