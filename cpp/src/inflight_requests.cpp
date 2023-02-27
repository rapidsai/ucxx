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

size_t InflightRequests::size() { return _inflightRequests->size(); }

void InflightRequests::insert(std::shared_ptr<Request> request)
{
  std::lock_guard<std::mutex> lock(_mutex);

  _inflightRequests->insert({request.get(), request});
}

void InflightRequests::merge(InflightRequestsMapPtr inflightRequestsMap)
{
  std::lock_guard<std::mutex> lock(_mutex);

  _inflightRequests->merge(*inflightRequestsMap);
}

void InflightRequests::remove(const Request* const request)
{
  do {
    int result = std::try_lock(_cancelMutex, _mutex);

    /**
     * If `result == -1` both locks have been acquired and it's safe to remove the
     * inflight request, otherwise retry.
     */
    if (result == -1) {
      auto search = _inflightRequests->find(request);
      if (search != _inflightRequests->end()) _inflightRequests->erase(search);
      _cancelMutex.unlock();
      _mutex.unlock();
      return;
    }
  } while (true);
}

size_t InflightRequests::cancelAll()
{
  // Fast path when no requests have been registered or the map has been
  // previously released.
  if (_inflightRequests->size() == 0) return 0;

  ucxx_debug("Canceling %lu requests", _inflightRequests->size());

  std::scoped_lock lock{_cancelMutex, _mutex};

  size_t total = _inflightRequests->size();

  for (auto& r : *_inflightRequests) {
    auto request = r.second;
    if (request != nullptr) { request->cancel(); }
  }
  _inflightRequests->clear();

  _cancelMutex.unlock();
  _mutex.unlock();

  return total;
}

InflightRequestsMapPtr InflightRequests::release()
{
  std::lock_guard<std::mutex> lock(_mutex);

  return std::exchange(_inflightRequests, std::make_unique<InflightRequestsMap>());
}

}  // namespace ucxx
