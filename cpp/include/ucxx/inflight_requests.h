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

typedef std::map<const Request* const, std::weak_ptr<Request>> InflightRequestsMap;
typedef std::unique_ptr<InflightRequestsMap> InflightRequestsMapPtr;

class InflightRequests {
 private:
  InflightRequestsMapPtr _inflightRequests{std::make_unique<InflightRequestsMap>()};
  std::mutex _mutex{};

 public:
  InflightRequests()                        = default;
  InflightRequests(const InflightRequests&) = delete;
  InflightRequests& operator=(InflightRequests const&) = delete;
  InflightRequests(InflightRequests&& o)               = delete;
  InflightRequests& operator=(InflightRequests&& o) = delete;

  ~InflightRequests();

  size_t size();

  void insert(std::shared_ptr<Request> request);

  void merge(InflightRequestsMapPtr request);

  void remove(const Request* const request);

  size_t cancelAll();

  InflightRequestsMapPtr release();
};

}  // namespace ucxx
