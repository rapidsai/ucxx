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
  std::scoped_lock lock(_mutex);

  _trackedRequests->_inflight->insert({request.get(), request});
}

void InflightRequests::merge(TrackedRequestsPtr trackedRequests)
{
  {
    std::scoped_lock lock(_mutex);
    _trackedRequests->_inflight->merge(*(trackedRequests->_inflight));
  }
  {
    std::scoped_lock lock(_cancelMutex);
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

void InflightRequests::remove(const Request* const request,
                              GenericCallbackUserFunction cancelInflightCallback)
{
  while (true) {
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

      size_t trackedRequestsCount =
        _trackedRequests->_inflight->size() + _trackedRequests->_canceling->size();

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
        _cancelMutex.unlock();
        return;
      } catch (const std::exception& e) {
        ucxx_warn("Exception in callback: %s", e.what());
        _cancelMutex.unlock();
        throw(e);
      }
    }
  }
}

size_t InflightRequests::getCancelingSize()
{
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

size_t InflightRequests::cancelAll(GenericCallbackUserFunction cancelInflightCallback)
{
  size_t total = 0;

  while (true) {
    // -1: both mutexes were locked.
    if (std::try_lock(_cancelMutex, _mutex) == -1) {
      auto total = _trackedRequests->_inflight->size();

      // Fast path when no requests have been registered or the map has been
      // previously released.
      if (total == 0) break;

      for (auto& r : *_trackedRequests->_inflight) {
        auto request = r.second;
        if (request != nullptr) {
          request->cancel();
          if (!request->isCompleted()) {
            _trackedRequests->_canceling->insert({request.get(), request});
          } else {
            auto status = request->getStatus();
          }
        }
      }
      _trackedRequests->_inflight->clear();

      break;
    }
  }

  size_t trackedRequestsCount =
    _trackedRequests->_inflight->size() + _trackedRequests->_canceling->size();

  /**
   * Unlock `_mutex` before calling the user callback to prevent deadlocks in case the
   * user callback happens to register another inflight request.
   */
  _mutex.unlock();
  try {
    if (cancelInflightCallback && trackedRequestsCount == 0) {
      ucxx_trace(
        "ucxx::InflightRequests::%s: %p, calling user cancel inflight callback", __func__, this);
      cancelInflightCallback();
    }
    _cancelMutex.unlock();
    return total;
  } catch (const std::exception& e) {
    ucxx_warn("Exception in callback: %s", e.what());
    _cancelMutex.unlock();
    throw(e);
  }
}

TrackedRequestsPtr InflightRequests::release()
{
  std::scoped_lock lock{_cancelMutex, _mutex};

  return std::exchange(_trackedRequests, std::make_unique<TrackedRequests>());
}

}  // namespace ucxx
