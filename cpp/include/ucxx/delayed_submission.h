/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/log.h>

namespace ucxx {

typedef std::function<void()> DelayedSubmissionCallbackType;

typedef std::shared_ptr<DelayedSubmissionCallbackType> DelayedSubmissionCallbackPtrType;

class DelayedSubmission {
 public:
  bool _send{false};       ///< Whether this is a send (`true`) operation or recv (`false`)
  void* _buffer{nullptr};  ///< Raw pointer to data buffer
  size_t _length{0};       ///< Length of the message in bytes
  ucp_tag_t _tag{0};       ///< Tag to match

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
   * @param[in] send    whether this is a send (`true`) or receive (`false`) operation.
   * @param[in] buffer  a raw pointer to the data being transferred.
   * @param[in] length  the size in bytes of the message being transfer.
   * @param[in] tag     tag to match for this operation (only applies for tag operations).
   */
  DelayedSubmission(const bool send, void* buffer, const size_t length, const ucp_tag_t tag = 0);
};

class DelayedSubmissionCollection {
 private:
  std::vector<DelayedSubmissionCallbackPtrType>
    _collection{};      ///< The collection of all known delayed submission operations.
  std::mutex _mutex{};  ///< Mutex to provide access to the collection.

 public:
  /**
   * @brief Default delayed submission collection constructor.
   *
   * Construct an empty collection of delayed submissions. Despite its name, a delayed
   * submission registration may be processed right after registration, thus effectively
   * making it an immediate submission.
   */
  DelayedSubmissionCollection()                                   = default;
  DelayedSubmissionCollection(const DelayedSubmissionCollection&) = delete;
  DelayedSubmissionCollection& operator=(DelayedSubmissionCollection const&) = delete;
  DelayedSubmissionCollection(DelayedSubmissionCollection&& o)               = delete;
  DelayedSubmissionCollection& operator=(DelayedSubmissionCollection&& o) = delete;

  /**
   * @brief Process all pending delayed submission operations.
   *
   * Process all pending delayed submissions and execute their callbacks. The execution
   * of the callbacks does not imply completion of the operation, only that it has been
   * submitted. The completion of each operation is handled externally by the
   * implementation of the object being processed, for example by checking the result
   * of `ucxx::Request::isCompleted()`.
   */
  void process();

  /**
   * @brief Register a request for delayed submission.
   *
   * Register a request for delayed submission with a callback that will be executed when
   * the request is in fact submitted when `process()` is called.
   *
   * @param[in] callback  the callback that will be executed by `process()` when the
   *                      operation is submitted.
   */
  void registerRequest(DelayedSubmissionCallbackType callback);
};

}  // namespace ucxx
