/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

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
 * Two container backends are available, selected at construction time via the
 * `UCXX_INFLIGHT_REQUESTS_BACKEND` environment variable:
 * - `vector` (default): best latency for small request counts, cache-friendly,
 *   no per-insert heap allocation, O(n) removal.
 * - `map`: O(1) amortized insert/remove, scales to thousands of concurrent
 *   inflight requests.
 */
class InflightRequests {
 private:
  bool _useMap{false};

  std::vector<std::shared_ptr<Request>> _inflightVec{};
  std::vector<std::shared_ptr<Request>> _cancelingVec{};

  std::unordered_map<Request*, std::shared_ptr<Request>> _inflightMap{};
  std::unordered_map<Request*, std::shared_ptr<Request>> _cancelingMap{};

  std::mutex _mutex{};

  void _doInsert(const std::shared_ptr<Request>& request);
  void _doRemove(const std::shared_ptr<Request>& request);
  size_t _doInflightSize() const;
  std::vector<std::shared_ptr<Request>> _doTakeInflight();
  void _doPutCanceling(std::vector<std::shared_ptr<Request>>* requests);
  size_t _doDropCanceled();
  size_t _doCancelingSize() const;
  void _doMergeInflight(std::vector<std::shared_ptr<Request>>* requests);
  void _doMergeCanceling(std::vector<std::shared_ptr<Request>>* requests);
  std::vector<std::shared_ptr<Request>> _doTakeCanceling();

 public:
  /**
   * @brief Construct with backend selected from UCXX_INFLIGHT_REQUESTS_BACKEND env var.
   *
   * Reads the `UCXX_INFLIGHT_REQUESTS_BACKEND` environment variable to select the
   * container backend. Valid values are `vector` (default) and `map`.
   */
  InflightRequests();

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
   * @param[in] request shared pointer to the request
   */
  void remove(const std::shared_ptr<Request>& request);

  /**
   * @brief Issue cancelation of all inflight requests and clear the internal container.
   *
   * Issue cancelation of all inflight requests known to this object and clear the
   * internal container. The total number of canceled requests is returned.
   *
   * @returns The total number of canceled requests.
   */
  size_t cancelAll();

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
};

}  // namespace ucxx
