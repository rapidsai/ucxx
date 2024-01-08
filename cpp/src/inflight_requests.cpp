/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>

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

void InflightRequests::merge(InflightRequestsMapPtrPair inflightRequestsMapPtrPair)
{
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _inflightRequests->merge(*(inflightRequestsMapPtrPair.first));
  }
  {
    std::lock_guard<std::mutex> lock(_cancelMutex);
    _cancelingRequests->merge(*(inflightRequestsMapPtrPair.second));
  }
}

void InflightRequests::remove(const Request* const request)
{
  do {
    int result = std::try_lock(_cancelMutex, _mutex);

    /**
     * `result` can be have one of three values:
     * -1 (both arguments were locked): Remove request and return.
     *  0 (failed to lock argument 0):  Failed acquiring `_cancelMutex`, cancel in
     *                                  progress, nothing to do but return. The method was
     *                                  called during execution of `cancelAll()` and the
     *                                  `Request*` callback was invoked.
     *  1 (failed to lock argument 1):  Only `_cancelMutex` was acquired, another
     *                                  operation in progress, retry.
     */
    if (result == 0) {
      return;
    } else if (result == -1) {
      auto search = _inflightRequests->find(request);
      decltype(search->second) tmpRequest;
      if (search != _inflightRequests->end()) {
        /**
         * If this is the last request to hold `std::shared_ptr<ucxx::Endpoint>` erasing it
         * may cause the `ucxx::Endpoint`s destructor and subsequently the `close()` method
         * to be called which will in turn call `cancelAll()` and attempt to take the
         * mutexes. For this reason we should make a temporary copy of the request being
         * erased from `_inflightRequests` to allow unlocking the mutexes and only then
         * destroy the object upon this method's return.
         */
        tmpRequest = search->second;
        _inflightRequests->erase(search);
      }
      _cancelMutex.unlock();
      _mutex.unlock();
      return;
    }
  } while (true);
}

size_t InflightRequests::dropCanceled()
{
  size_t removed = 0;

  {
    std::scoped_lock lock{_cancelMutex};
    for (auto it = _cancelingRequests->begin(); it != _cancelingRequests->end();) {
      auto request = it->second;
      if (request != nullptr && request->getStatus() != UCS_INPROGRESS) {
        it = _cancelingRequests->erase(it);
        ++removed;
      } else {
        ++it;
      }
    }
  }

  return removed;
}

size_t InflightRequests::getCancelingCount()
{
  dropCanceled();
  size_t cancelingCount = 0;
  {
    std::scoped_lock lock{_cancelMutex};
    cancelingCount = _cancelingRequests->size();
  }

  return cancelingCount;
}

size_t InflightRequests::cancelAll()
{
  decltype(_inflightRequests) toCancel;
  size_t total;
  {
    std::scoped_lock lock{_cancelMutex, _mutex};
    total = _inflightRequests->size();

    // Fast path when no requests have been registered or the map has been
    // previously released.
    if (total == 0) return 0;

    toCancel = std::exchange(_inflightRequests, std::make_unique<InflightRequestsMap>());
  }

  ucxx_debug("Canceling %lu requests", total);

  for (auto& r : *toCancel) {
    auto request = r.second;
    if (request != nullptr) { request->cancel(); }
  }

  {
    std::scoped_lock lock{_cancelMutex, _mutex};
    _cancelingRequests->merge(*toCancel);
  }
  dropCanceled();

  return total;
}

InflightRequestsMapPtrPair InflightRequests::release()
{
  std::scoped_lock lock{_cancelMutex, _mutex};

  return {std::exchange(_inflightRequests, std::make_unique<InflightRequestsMap>()),
          std::exchange(_cancelingRequests, std::make_unique<InflightRequestsMap>())};
}

}  // namespace ucxx
