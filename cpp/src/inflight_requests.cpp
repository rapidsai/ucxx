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

size_t InflightRequests::size() { return _trackedRequests->_inflight->size(); }

void InflightRequests::insert(std::shared_ptr<Request> request)
{
  std::lock_guard<std::mutex> lock(_mutex);

  _trackedRequests->_inflight->insert({request.get(), request});
}

void InflightRequests::merge(TrackedRequestsPtr trackedRequests)
{
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _trackedRequests->_inflight->merge(*(trackedRequests->_inflight));
  }
  {
    std::lock_guard<std::mutex> lock(_cancelMutex);
    _trackedRequests->_canceling->merge(*(trackedRequests->_canceling));
  }
}

static void findAndRemove(InflightRequestsMap* requestsMap, const Request* const request)
{
  auto search = requestsMap->find(request);
  decltype(search->second) tmpRequest;
  if (search != requestsMap->end()) {
    /**
     * If this is the last request to hold `std::shared_ptr<ucxx::Endpoint>` erasing it
     * may cause the `ucxx::Endpoint`s destructor and subsequently the `closeBlocking()`
     * method to be called which will in turn call `cancelAll()` and attempt to take the
     * mutexes. For this reason we should make a temporary copy of the request being
     * erased from `_trackedRequests->_inflight` to allow unlocking the mutexes and only then
     * destroy the object upon this method's return.
     */
    tmpRequest = search->second;
    requestsMap->erase(search);
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
      findAndRemove(_trackedRequests->_inflight.get(), request);
      findAndRemove(_trackedRequests->_canceling.get(), request);
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
    for (auto it = _trackedRequests->_canceling->begin();
         it != _trackedRequests->_canceling->end();) {
      auto request = it->second;
      if (request != nullptr && request->getStatus() != UCS_INPROGRESS) {
        it = _trackedRequests->_canceling->erase(it);
        ++removed;
      } else {
        ++it;
      }
    }
  }

  return removed;
}

size_t InflightRequests::getCancelingSize()
{
  // dropCanceled();
  size_t cancelingSize = 0;
  {
    std::scoped_lock lock{_cancelMutex};
    cancelingSize = _trackedRequests->_canceling->size();
  }

  return cancelingSize;
}

size_t InflightRequests::getInflightSize()
{
  size_t inflightSize = 0;
  {
    std::scoped_lock lock{_mutex};
    inflightSize = _trackedRequests->_inflight->size();
  }

  return inflightSize;
}

size_t InflightRequests::cancelAll()
{
  std::scoped_lock lock{_cancelMutex, _mutex};
  auto total = _trackedRequests->_inflight->size();

  // Fast path when no requests have been registered or the map has been
  // previously released.
  if (total == 0) return 0;

  for (auto& r : *_trackedRequests->_inflight) {
    auto request = r.second;
    if (request != nullptr) {
      request->cancel();
      if (!request->isCompleted()) _trackedRequests->_canceling->insert({request.get(), request});
    }
  }
  _trackedRequests->_inflight->clear();

  // dropCanceled();

  return total;
}

TrackedRequestsPtr InflightRequests::release()
{
  std::scoped_lock lock{_cancelMutex, _mutex};

  return std::exchange(_trackedRequests, std::make_unique<TrackedRequests>());
}

}  // namespace ucxx
