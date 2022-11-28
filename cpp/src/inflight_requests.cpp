/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <ucxx/inflight_requests.h>
#include <ucxx/log.h>
#include <ucxx/request.h>

namespace ucxx {

InflightRequests::~InflightRequests() { cancelAll(); }

size_t InflightRequests::size()
{
  if (_inflightRequests == nullptr) return 0;

  return _inflightRequests->size();
}

void InflightRequests::insert(std::shared_ptr<Request> request)
{
  std::weak_ptr<Request> weakReq = request;

  std::lock_guard<std::mutex> lock(_mutex);

  _inflightRequests->insert({request.get(), weakReq});
}

void InflightRequests::merge(InflightRequestsMapPtr inflightRequestsMap)
{
  std::lock_guard<std::mutex> lock(_mutex);

  _inflightRequests->merge(*inflightRequestsMap);
}

void InflightRequests::remove(const Request* const request)
{
  std::lock_guard<std::mutex> lock(_mutex);

  auto search = _inflightRequests->find(request);
  if (search != _inflightRequests->end()) _inflightRequests->erase(search);
}

size_t InflightRequests::cancelAll()
{
  // Fast path when no requests have been registered or the map has been
  // previously released.
  if (_inflightRequests->size() == 0) return 0;

  ucxx_debug("Canceling %lu requests", _inflightRequests->size());

  std::lock_guard<std::mutex> lock(_mutex);

  size_t total = _inflightRequests->size();

  for (auto& r : *_inflightRequests) {
    if (auto request = r.second.lock()) { request->cancel(); }
  }
  _inflightRequests->clear();

  return total;
}

InflightRequestsMapPtr InflightRequests::release()
{
  std::lock_guard<std::mutex> lock(_mutex);

  return std::exchange(_inflightRequests, std::make_unique<InflightRequestsMap>());
}

}  // namespace ucxx
