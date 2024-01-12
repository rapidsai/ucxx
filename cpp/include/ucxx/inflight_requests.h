/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <utility>

namespace ucxx {

class Request;

typedef std::map<const Request* const, std::shared_ptr<Request>> InflightRequestsMap;
typedef std::unique_ptr<InflightRequestsMap> InflightRequestsMapPtr;
typedef struct TrackedRequests {
  InflightRequestsMapPtr _inflight;
  InflightRequestsMapPtr _canceling;

  TrackedRequests()
    : _inflight(std::make_unique<InflightRequestsMap>()),
      _canceling(std::make_unique<InflightRequestsMap>())
  {
  }
} TrackedRequests;
typedef std::unique_ptr<TrackedRequests> TrackedRequestsPtr;

class InflightRequests {
 private:
  TrackedRequestsPtr _trackedRequests{
    std::make_unique<TrackedRequests>()};  ///< Container storing pointers to all inflight
                                           ///< and in cancelation process requests known to
                                           ///< the owner of this object
  std::mutex _mutex{};  ///< Mutex to control access to inflight requests container
  std::mutex
    _cancelMutex{};  ///< Mutex to allow cancelation and prevent removing requests simultaneously

  /**
   * @brief Drop references to requests that completed cancelation.
   *
   * Drops references to requests that completed cancelation and stop tracking them.
   *
   * @returns The number of requests that have completed cancelation since last call.
   */
  size_t dropCanceled();

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
  size_t size();

  /**
   * @brief Insert an inflight requests to the container.
   *
   * @param[in] request a `std::shared_ptr<Request>` with the inflight request.
   */
  void insert(std::shared_ptr<Request> request);

  /**
   * @brief Merge containers of inflight requests with the internal containers.
   *
   * Merge containers of inflight requests obtained from `InflightRequests::release()` of
   * another object with the internal containers.
   *
   * @param[in] trackedRequests containers of tracked inflight requests to merge with the
   *                            internal tracked inflight requests.
   */
  void merge(TrackedRequestsPtr trackedRequests);

  /**
   * @brief Remove an inflight request from the internal container.
   *
   * Remove the reference to a specific request from the internal container. This should
   * be called when a request has completed and the `InflightRequests` owner does not need
   * to keep track of it anymore. The raw pointer to a `ucxx::Request` is passed here as
   * opposed to the usual `std::shared_ptr<ucxx::Request>` used elsewhere, this is because
   * the raw pointer address is used as key to the requests reference, and this is called
   * called from the object's destructor.
   *
   * @param[in] request raw pointer to the request
   */
  void remove(const Request* const request);

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
   * @returns The internally-tracked containers.
   */
  TrackedRequestsPtr release();

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
  size_t getCancelingSize();
};

}  // namespace ucxx
