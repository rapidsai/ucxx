/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>

#include <ucxx/inflight_requests.h>
#include <ucxx/log.h>
#include <ucxx/request.h>

namespace ucxx {

InflightRequests::~InflightRequests() { cancelAll(); }

size_t InflightRequests::size()
{
  std::scoped_lock localLock{_mutex};
  std::lock_guard<std::mutex> lock(_trackedRequests->_mutex);
  return _trackedRequests->_inflight.size();
}

void InflightRequests::insert(std::shared_ptr<Request> request)
{
  std::scoped_lock localLock{_mutex};
  std::lock_guard<std::mutex> lock(_trackedRequests->_mutex);

  _trackedRequests->_inflight.insert({request, request});
}

void InflightRequests::merge(TrackedRequestsPtr trackedRequests)
{
  {
    if (trackedRequests == nullptr) return;

    std::scoped_lock localLock{_mutex};
    std::scoped_lock lock{_trackedRequests->_cancelMutex,
                          _trackedRequests->_mutex,
                          trackedRequests->_cancelMutex,
                          trackedRequests->_mutex};

    _trackedRequests->_inflight.merge(trackedRequests->_inflight);
    _trackedRequests->_canceling.merge(trackedRequests->_canceling);
  }
}

void InflightRequests::remove(std::shared_ptr<Request> request)
{
  do {
    std::scoped_lock localLock{_mutex};
    int result = std::try_lock(_trackedRequests->_cancelMutex, _trackedRequests->_mutex);

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
      auto search = _trackedRequests->_inflight.find(request);
      decltype(search->second) tmpRequest;
      if (search != _trackedRequests->_inflight.end()) {
        /**
         * If this is the last request to hold `std::shared_ptr<ucxx::Endpoint>` erasing it
         * may cause the `ucxx::Endpoint`s destructor and subsequently the `closeBlocking()`
         * method to be called which will in turn call `cancelAll()` and attempt to take the
         * mutexes. For this reason we should make a temporary copy of the request being
         * erased from `_trackedRequests->_inflight` to allow unlocking the mutexes and only then
         * destroy the object upon this method's return.
         */
        tmpRequest = search->second;
        _trackedRequests->_inflight.erase(search);
      }
      _trackedRequests->_cancelMutex.unlock();
      _trackedRequests->_mutex.unlock();
      return;
    }
  } while (true);
}

size_t InflightRequests::dropCanceled()
{
  size_t removed = 0;

  {
    std::scoped_lock localLock{_mutex};
    std::scoped_lock lock{_trackedRequests->_cancelMutex};
    for (auto it = _trackedRequests->_canceling.begin();
         it != _trackedRequests->_canceling.end();) {
      auto request = it->second;
      if (request != nullptr && request->getStatus() != UCS_INPROGRESS) {
        it = _trackedRequests->_canceling.erase(it);
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
  dropCanceled();
  size_t cancelingSize = 0;
  {
    std::scoped_lock localLock{_mutex};
    std::scoped_lock lock{_trackedRequests->_cancelMutex};
    cancelingSize = _trackedRequests->_canceling.size();
  }

  return cancelingSize;
}

size_t InflightRequests::cancelAll()
{
  decltype(_trackedRequests->_inflight) toCancel;
  size_t total;
  {
    std::scoped_lock localLock{_mutex};
    std::scoped_lock lock{_trackedRequests->_cancelMutex, _trackedRequests->_mutex};
    total = _trackedRequests->_inflight.size();

    // Fast path when no requests have been registered or the map has been
    // previously released.
    if (total == 0) return 0;

    toCancel = std::exchange(_trackedRequests->_inflight, InflightRequestsMap());

    ucxx_debug("ucxx::InflightRequests::%s, canceling %lu requests", __func__, total);

    for (auto& r : toCancel) {
      auto request = r.second;
      if (request != nullptr) { request->cancel(); }
    }

    _trackedRequests->_canceling.merge(toCancel);

    // Drop canceled requests. Do not call `dropCanceled()` to prevent locking mutexes
    // again.
    for (auto it = _trackedRequests->_canceling.begin();
         it != _trackedRequests->_canceling.end();) {
      auto request = it->second;
      if (request != nullptr && request->getStatus() != UCS_INPROGRESS) {
        it = _trackedRequests->_canceling.erase(it);
      } else {
        ++it;
      }
    }
  }

  return total;
}

TrackedRequestsPtr InflightRequests::release()
{
  std::scoped_lock localLock{_mutex};
  std::scoped_lock lock{_trackedRequests->_cancelMutex, _trackedRequests->_mutex};

  return std::exchange(_trackedRequests, std::make_unique<TrackedRequests>());
}

}  // namespace ucxx
