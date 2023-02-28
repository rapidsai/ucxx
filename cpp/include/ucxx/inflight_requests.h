/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <map>
#include <memory>
#include <mutex>

namespace ucxx {

class Request;

typedef std::map<const Request* const, std::shared_ptr<Request>> InflightRequestsMap;
typedef std::unique_ptr<InflightRequestsMap> InflightRequestsMapPtr;

class InflightRequests {
 private:
  InflightRequestsMapPtr _inflightRequests{
    std::make_unique<InflightRequestsMap>()};  ///< Container storing pointers to all inflight
                                               ///< requests known to the owner of this object
  std::mutex _mutex{};  ///< Mutex to control access to inflight requests container
  std::mutex
    _cancelMutex{};  ///< Mutex to allow cancelation and prevent removing requests simultaneously

 public:
  /**
   * @brief Default constructor.
   */
  InflightRequests() = default;

  InflightRequests(const InflightRequests&) = delete;
  InflightRequests& operator=(InflightRequests const&) = delete;
  InflightRequests(InflightRequests&& o)               = delete;
  InflightRequests& operator=(InflightRequests&& o) = delete;

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
   * @brief Merge a container of inflight requests with the internal container.
   *
   * Merge a container of inflight requests obtained from `InflightRequests::release()` of
   * another object with the internal container.
   *
   * @param[in] inflightRequestsMap container of inflight requests to merge with the
   *                                internal container.
   */
  void merge(InflightRequestsMapPtr inflightRequestsMap);

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
   * @brief Releases the internal container.
   *
   * Releases the internal container that can be merged into another `InflightRequests`
   * object with `InflightRequests::release()`. Effectively leaves the internal state as a
   * clean, new object.
   *
   * @returns The internal container.
   */
  InflightRequestsMapPtr release();
};

}  // namespace ucxx
