/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include <ucxx/typedefs.h>

namespace ucxx {

class Request;

/**
 * @brief Container for transferring tracked requests between InflightRequests instances.
 *
 * Used by `InflightRequests::release()` and `InflightRequests::merge()` to move
 * request ownership between instances (e.g., from an endpoint to the worker during
 * endpoint close).
 */
struct TrackedRequests {
  std::vector<std::shared_ptr<Request>> inflight{};   ///< Valid requests awaiting completion.
  std::vector<std::shared_ptr<Request>> canceling{};  ///< Requests scheduled for cancelation.
};

/**
 * @brief Handle tracked requests.
 *
 * Handle tracked requests, providing functionality so that its owner can modify those
 * requests, performing operations such as insertion, removal and cancelation.
 *
 * Uses `std::unordered_set<shared_ptr<Request>>` for O(1) amortized insert/remove that
 * scales to thousands of concurrent inflight requests.
 */
class InflightRequests {
 private:
  std::unordered_set<std::shared_ptr<Request>> _inflight{};
  std::unordered_set<std::shared_ptr<Request>> _canceling{};

  std::mutex _mutex{};
  std::atomic<bool> _cancelAllInProgress{false};

 public:
  /**
   * @brief Default constructor.
   */
  InflightRequests() = default;

  InflightRequests(const InflightRequests&)            = delete;
  InflightRequests& operator=(InflightRequests const&) = delete;
  InflightRequests(InflightRequests&& o)               = delete;
  InflightRequests& operator=(InflightRequests&& o)    = delete;

  /**
   * @brief Destructor.
   *
   * Cancels all inflight requests before destruction.
   */
  ~InflightRequests();

  /**
   * @brief Query the number of pending inflight requests.
   *
   * @returns The number of pending inflight requests.
   */
  [[nodiscard]] size_t size();

  /**
   * @brief Insert an inflight request into the container.
   *
   * @param[in] request a `std::shared_ptr<Request>` with the inflight request.
   */
  void insert(const std::shared_ptr<Request>& request);

  /**
   * @brief Merge containers of inflight requests with the internal containers.
   *
   * Merge containers of inflight requests obtained from `InflightRequests::release()` of
   * another object with the internal containers.
   *
   * @param[in] trackedRequests containers of tracked inflight requests to merge with the
   *                            internal tracked inflight requests.
   */
  void merge(TrackedRequests&& trackedRequests);

  /**
   * @brief Remove an inflight request from the internal container.
   *
   * Remove the reference to a specific request from the internal container. This should
   * be called when a request has completed and the `InflightRequests` owner does not need
   * to keep track of it anymore.
   *
   * Supports an optional callback function to be called exclusively if there are no
   * more requests inflight or canceling. Be advised that before the callback is called the
   * mutex that controls inflight requests is released to prevent deadlocks in case the
   * callback happens to register a new inflight request, therefore there's no guarantee
   * that another inflight request won't be registered between the time in which the mutex
   * is released and the callback is executed.
   *
   * @param[in] request shared pointer to the request
   * @param[in] callbackFunction  function to be called upon termination and only if no
   *                              further requests inflight or canceling remain.
   */
  void remove(const std::shared_ptr<Request>& request,
              VoidCallbackUserFunction callbackFunction = nullptr);

  /**
   * @brief Issue cancelation of all inflight requests and clear the internal container.
   *
   * Issue cancelation of all inflight requests known to this object and clear the
   * internal container. The total number of canceled requests is returned.
   *
   * Supports an optional callback function to be called exclusively if there are no
   * more requests inflight or canceling. Be advised that before the callback is called the
   * mutex that controls inflight requests is released to prevent deadlocks in case the
   * callback happens to register a new inflight request, therefore there's no guarantee
   * that another inflight request won't be registered between the time in which the mutex
   * is released and the callback is executed.
   *
   * @param[in] callbackFunction  function to be called upon termination and only if no
   *                              further requests inflight or canceling remain.
   *
   * @returns The total number of canceled requests.
   */
  size_t cancelAll(VoidCallbackUserFunction callbackFunction = nullptr);

  /**
   * @brief Releases the internally-tracked containers.
   *
   * Releases the internally-tracked containers that can be merged into another
   * `InflightRequests` object with `InflightRequests::merge()`. Effectively leaves the
   * internal state as a clean, new object.
   *
   * @returns The tracked requests as vectors for transfer.
   */
  [[nodiscard]] TrackedRequests release();

  /**
   * @brief Get count of requests in process of cancelation.
   *
   * After `cancelAll()` is called the requests are scheduled for cancelation but may not
   * complete immediately, therefore they are tracked until cancelation is completed. This
   * method returns the count of requests in process of cancelation and drops references
   * to those that have completed.
   *
   * @returns The count of requests that are in process of cancelation.
   */
  [[nodiscard]] size_t getCancelingSize();

  /**
   * @brief Get count of inflight requests.
   *
   * Get the count of inflight requests that have not yet completed nor have been scheduled
   * for cancelation.
   *
   * @returns The count of inflight requests.
   */
  [[nodiscard]] size_t getInflightSize();
};

}  // namespace ucxx
