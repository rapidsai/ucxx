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

  _trackedRequests->_inflight.insert({request.get(), request});
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

static std::unique_ptr<InflightRequestsMap> findAndRemove(InflightRequestsMap* requestsMap,
                                                          const Request* const request)
{
  auto removed = std::make_unique<InflightRequestsMap>();
  auto search  = requestsMap->find(request);
  if (search != requestsMap->end()) {
    /**
     * If this is the last request to hold `std::shared_ptr<ucxx::Endpoint>` erasing it
     * may cause the `ucxx::Endpoint`s destructor and subsequently the `closeBlocking()`
     * method to be called which will in turn call `cancelAll()` and attempt to take the
     * mutexes. For this reason we should make a temporary copy of the request being
     * erased from `_trackedRequests->_inflight` in `removed` to allow the caller to unlock
     * the mutexes and only then destroy the object.
     */
    removed->insert({request, search->second});

    requestsMap->erase(search);
  }

  return removed;
}

void InflightRequests::remove(const Request* const request,
                              GenericCallbackUserFunction cancelInflightCallback)
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
      /**
       * Retain references to removed pointers to prevent their refcounts from going to
       * while locks are held, which may trigger a chain effect and cause `this` itself
       * from destroying and thus call `cancelAll()` which will then cause a deadlock.
       */
      auto removedInflight  = findAndRemove(&_trackedRequests->_inflight, request);
      auto removedCanceling = findAndRemove(&_trackedRequests->_canceling, request);

      size_t trackedRequestsCount =
        _trackedRequests->_inflight.size() + _trackedRequests->_canceling.size();

      /**
       * Unlock `_mutex` before calling the user callback to prevent deadlocks in case the
       * user callback happens to register another inflight request.
       */
      _mutex.unlock();
      try {
        if (cancelInflightCallback && trackedRequestsCount == 0) {
          ucxx_trace("ucxx::InflightRequests::%s: %p, calling user cancel inflight callback",
                     __func__,
                     this);
          cancelInflightCallback();
        }
        _trackedRequests->_cancelMutex.unlock();
        return;
      } catch (const std::exception& e) {
        ucxx_warn("Exception in callback: %s", e.what());
        _trackedRequests->_cancelMutex.unlock();
        throw(e);
      }
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
  size_t cancelingSize = 0;
  {
    std::scoped_lock localLock{_mutex};
    std::scoped_lock lock{_trackedRequests->_cancelMutex};
    cancelingSize = _trackedRequests->_canceling.size();
  }

  return cancelingSize;
}

size_t InflightRequests::getInflightSize()
{
  size_t inflightSize = 0;
  {
    std::scoped_lock lock{_mutex};
    inflightSize = _trackedRequests->_inflight.size();
  }

  return inflightSize;
}

size_t InflightRequests::cancelAll(GenericCallbackUserFunction cancelInflightCallback)
{
  size_t total = 0;

  while (true) {
    // -1: both mutexes were locked.
    if (std::try_lock(_mutex, _trackedRequests->_cancelMutex, _trackedRequests->_mutex) == -1) {
      auto total = _trackedRequests->_inflight.size();

      // Fast path when no requests have been registered or the map has been
      // previously released.
      if (total == 0) break;

      auto toCancel = std::exchange(_trackedRequests->_inflight, InflightRequestsMap());

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

      break;
    }
  }

  size_t trackedRequestsCount =
    _trackedRequests->_inflight.size() + _trackedRequests->_canceling.size();

  /**
   * Unlock `_mutex` before calling the user callback to prevent deadlocks in case the
   * user callback happens to register another inflight request.
   */
  _trackedRequests->_mutex.unlock();
  try {
    if (cancelInflightCallback && trackedRequestsCount == 0) {
      ucxx_trace(
        "ucxx::InflightRequests::%s: %p, calling user cancel inflight callback", __func__, this);
      cancelInflightCallback();
    }
    _trackedRequests->_cancelMutex.unlock();
    _mutex.unlock();
    return total;
  } catch (const std::exception& e) {
    ucxx_warn("Exception in callback: %s", e.what());
    _trackedRequests->_cancelMutex.unlock();
    _mutex.unlock();
    throw(e);
  }
}

TrackedRequestsPtr InflightRequests::release()
{
  std::scoped_lock localLock{_mutex};
  std::scoped_lock lock{_trackedRequests->_cancelMutex, _trackedRequests->_mutex};

  return std::exchange(_trackedRequests, std::make_unique<TrackedRequests>());
}

}  // namespace ucxx
