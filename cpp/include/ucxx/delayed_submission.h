/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/log.h>

namespace ucxx {

typedef std::function<void()> DelayedSubmissionCallbackType;

class DelayedSubmission {
 public:
  bool _send{false};       ///< Whether this is a send (`true`) operation or recv (`false`)
  void* _buffer{nullptr};  ///< Raw pointer to data buffer
  size_t _length{0};       ///< Length of the message in bytes
  ucp_tag_t _tag{0};       ///< Tag to match
  ucs_memory_type_t _memoryType{UCS_MEMORY_TYPE_UNKNOWN};  ///< Buffer memory type

  DelayedSubmission() = delete;

  /**
   * @brief Constructor for a delayed submission operation.
   *
   * Construct a delayed submission operation. Delayed submission means that a transfer
   * operation will not be submitted immediately, but will rather be delayed for the next
   * progress iteration.
   *
   * This may be useful to avoid any transfer operations to be executed directly in the
   * application thread, delaying all of them for the worker progress thread when enabled.
   * With this approach any perceived overhead will be removed from the application thread,
   * and thus provide some speedup in certain situations. It may be also useful to prevent
   * a multi-threaded application for blocking while waiting for the UCX spinlock, since
   * all transfer operations may be pushed to the worker progress thread.
   *
   * @param[in] send        whether this is a send (`true`) or receive (`false`) operation.
   * @param[in] buffer      a raw pointer to the data being transferred.
   * @param[in] length      the size in bytes of the message being transfer.
   * @param[in] tag         tag to match for this operation (only applies for tag
   *                        operations).
   * @param[in] memoryType  the memory type of the buffer.
   */
  DelayedSubmission(const bool send,
                    void* buffer,
                    const size_t length,
                    const ucp_tag_t tag                = 0,
                    const ucs_memory_type_t memoryType = UCS_MEMORY_TYPE_UNKNOWN);
};

template <typename T>
class BaseDelayedSubmissionCollection {
 protected:
  std::string_view _name{
    "undefined"};                ///< The human-readable name of the collection, used for logging
  std::vector<T> _collection{};  ///< The collection.
  std::mutex _mutex{};           ///< Mutex to provide access to `_collection`.

  /**
   * @brief Log message during `schedule()`.
   *
   * Log a specialized message while `schedule()` is being executed.
   *
   * @param[in] item      the callback that was passed as argument to `schedule()`.
   */
  virtual void scheduleLog(T item) = 0;

  /**
   * @brief Process a single item during `process()`.
   *
   * Method called by `process()` to process a single item of the collection.
   *
   * @param[in] item      the callback that was passed as argument to `schedule()` when
   *                      the first registered.
   */
  virtual void processItem(T item) = 0;

 public:
  /**
   * @brief Constructor for a thread-safe delayed submission collection.
   *
   * Construct a thread-safe delayed submission collection. A delayed submission collection
   * provides two operations: schedule and process. The `schedule()` method will push an
   * operation into the collection, whereas the `process()` will invoke all callbacks that
   * were previously pushed into the collection and clear the collection.
   *
   * @param[in] name  human-readable name of the collection, used for logging.
   */
  explicit BaseDelayedSubmissionCollection(const std::string_view name) : _name{name} {}

  BaseDelayedSubmissionCollection()                                                  = delete;
  BaseDelayedSubmissionCollection(const BaseDelayedSubmissionCollection&)            = delete;
  BaseDelayedSubmissionCollection& operator=(BaseDelayedSubmissionCollection const&) = delete;
  BaseDelayedSubmissionCollection(BaseDelayedSubmissionCollection&& o)               = delete;
  BaseDelayedSubmissionCollection& operator=(BaseDelayedSubmissionCollection&& o)    = delete;

  /**
   * @brief Register a callable or complex-type for delayed submission.
   *
   * Register a simple callback, or complex-type with a callback (requires specialization),
   * for delayed submission that will be executed when the request is in fact submitted when
   * `process()` is called.
   *
   * @throws std::runtime_error if delayed request submission was disabled at construction.
   *
   * @param[in] item      the callback that will be executed by `process()` when the
   *                      operation is submitted.
   */
  virtual void schedule(T item, const bool contextInitialized)
  {
    if (!contextInitialized) throw std::runtime_error("Context not initialized");

    {
      std::lock_guard<std::mutex> lock(_mutex);
      _collection.push_back(item);
    }
    scheduleLog(item);
  }

  /**
   * @brief Process all pending callbacks.
   *
   * Process all pending generic. Generic callbacks are deemed completed when their
   * execution completes.
   */
  void process()
  {
    decltype(_collection) itemsToProcess;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      // Move _collection to a local copy in order to to hold the lock for as
      // short as possible
      itemsToProcess = std::move(_collection);
    }

    if (itemsToProcess.size() > 0) {
      ucxx_trace_req("Submitting %lu %s callbacks", itemsToProcess.size(), _name);
      for (auto& item : itemsToProcess)
        processItem(item);
    }
  }
};

class RequestDelayedSubmissionCollection
  : public BaseDelayedSubmissionCollection<
      std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType>> {
 protected:
  void scheduleLog(
    std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item) override;

  void processItem(
    std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item) override;

 public:
  explicit RequestDelayedSubmissionCollection(const std::string_view name);
  RequestDelayedSubmissionCollection()                                                     = delete;
  RequestDelayedSubmissionCollection(const RequestDelayedSubmissionCollection&)            = delete;
  RequestDelayedSubmissionCollection& operator=(RequestDelayedSubmissionCollection const&) = delete;
  RequestDelayedSubmissionCollection(RequestDelayedSubmissionCollection&& o)               = delete;
  RequestDelayedSubmissionCollection& operator=(RequestDelayedSubmissionCollection&& o)    = delete;
};

class GenericDelayedSubmissionCollection
  : public BaseDelayedSubmissionCollection<DelayedSubmissionCallbackType> {
 protected:
  void scheduleLog(DelayedSubmissionCallbackType item) override;

  void processItem(DelayedSubmissionCallbackType callback) override;

 public:
  explicit GenericDelayedSubmissionCollection(const std::string_view name);
  GenericDelayedSubmissionCollection()                                                     = delete;
  GenericDelayedSubmissionCollection(const GenericDelayedSubmissionCollection&)            = delete;
  GenericDelayedSubmissionCollection& operator=(GenericDelayedSubmissionCollection const&) = delete;
  GenericDelayedSubmissionCollection(GenericDelayedSubmissionCollection&& o)               = delete;
  GenericDelayedSubmissionCollection& operator=(GenericDelayedSubmissionCollection&& o)    = delete;
};

class DelayedSubmissionCollection {
 private:
  GenericDelayedSubmissionCollection _genericPre{
    "generic pre"};  ///< The collection of all known generic pre-progress operations.
  GenericDelayedSubmissionCollection _genericPost{
    "generic post"};  ///< The collection of all known generic post-progress operations.
  RequestDelayedSubmissionCollection _requests{
    "request"};  ///< The collection of all known delayed request submission operations.
  bool _enableDelayedRequestSubmission{false};

 public:
  /**
   * @brief Default delayed submission collection constructor.
   *
   * Construct an empty collection of delayed submissions. Despite its name, a delayed
   * submission registration may be processed right after registration, thus effectively
   * making it an immediate submission.
   *
   * @param[in] enableDelayedRequestSubmission  whether request submission should be
   *                                            enabled, if `false`, only generic
   *                                            callbacks are enabled.
   */
  explicit DelayedSubmissionCollection(bool enableDelayedRequestSubmission = false);

  DelayedSubmissionCollection()                                              = delete;
  DelayedSubmissionCollection(const DelayedSubmissionCollection&)            = delete;
  DelayedSubmissionCollection& operator=(DelayedSubmissionCollection const&) = delete;
  DelayedSubmissionCollection(DelayedSubmissionCollection&& o)               = delete;
  DelayedSubmissionCollection& operator=(DelayedSubmissionCollection&& o)    = delete;

  /**
   * @brief Process pending delayed request submission and generic-pre callback operations.
   *
   * Process all pending delayed request submissions and generic callbacks. Generic
   * callbacks are deemed completed when their execution completes. On the other hand, the
   * execution of the delayed request submission callbacks does not imply completion of the
   * operation, only that it has been submitted. The completion of each delayed request
   * submission is handled externally by the implementation of the object being processed,
   * for example by checking the result of `ucxx::Request::isCompleted()`.
   */
  void processPre();

  /**
   * @brief Process all pending generic-post callback operations.
   *
   * Process all pending generic-post callbacks. Generic callbacks are deemed completed when
   * their execution completes.
   */
  void processPost();

  /**
   * @brief Register a request for delayed submission.
   *
   * Register a request for delayed submission with a callback that will be executed when
   * the request is in fact submitted when `processPre()` is called.
   *
   * @throws std::runtime_error if delayed request submission was disabled at construction.
   *
   * @param[in] request   the request to which the callback belongs, ensuring it remains
   *                      alive until the callback is invoked.
   * @param[in] callback  the callback that will be executed by `processPre()` when the
   *                      operation is submitted.
   */
  void registerRequest(std::shared_ptr<Request> request, DelayedSubmissionCallbackType callback);

  /**
   * @brief Register a generic callback to execute during `processPre()`.
   *
   * Register a generic callback that will be executed when `processPre()` is called.
   * Lifetime of the callback must be ensured by the caller.
   *
   * @param[in] callback  the callback that will be executed by `processPre()`.
   */
  void registerGenericPre(DelayedSubmissionCallbackType callback);

  /**
   * @brief Register a generic callback to execute during `processPost()`.
   *
   * Register a generic callback that will be executed when `processPost()` is called.
   * Lifetime of the callback must be ensured by the caller.
   *
   * @param[in] callback  the callback that will be executed by `processPre()`.
   */
  void registerGenericPost(DelayedSubmissionCallbackType callback);

  /**
   * @brief Inquire if delayed request submission is enabled.
   *
   * Check whether delayed submission request is enabled, in which case `registerRequest()`
   * may be used to register requests that will be executed during `processPre()`.
   *
   * @returns `true` if a delayed request submission is enabled, `false` otherwise.
   */
  bool isDelayedRequestSubmissionEnabled() const;
};

}  // namespace ucxx
