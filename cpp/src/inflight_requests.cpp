/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ucxx/inflight_requests.h>
#include <ucxx/log.h>
#include <ucxx/request.h>

namespace ucxx {

// ---- Private backend helpers ------------------------------------------------

void InflightRequests::_doInsert(const std::shared_ptr<Request>& request)
{
  if (_useMap)
    _inflightMap.emplace(request.get(), request);
  else
    _inflightVec.push_back(request);
}

void InflightRequests::_doRemove(const std::shared_ptr<Request>& request)
{
  if (_useMap) {
    _inflightMap.erase(request.get());
  } else {
    auto it = std::find(_inflightVec.begin(), _inflightVec.end(), request);
    if (it != _inflightVec.end()) {
      *it = std::move(_inflightVec.back());
      _inflightVec.pop_back();
    }
  }
}

size_t InflightRequests::_doInflightSize() const
{
  return _useMap ? _inflightMap.size() : _inflightVec.size();
}

std::vector<std::shared_ptr<Request>> InflightRequests::_doTakeInflight()
{
  std::vector<std::shared_ptr<Request>> result;
  if (_useMap) {
    result.reserve(_inflightMap.size());
    for (auto& kv : _inflightMap)
      result.push_back(std::move(kv.second));
    _inflightMap.clear();
  } else {
    result = std::exchange(_inflightVec, {});
  }
  return result;
}

void InflightRequests::_doPutCanceling(std::vector<std::shared_ptr<Request>>* requests)
{
  if (_useMap) {
    for (auto& r : *requests)
      if (r) _cancelingMap.emplace(r.get(), std::move(r));
  } else {
    _cancelingVec.insert(_cancelingVec.end(),
                         std::make_move_iterator(requests->begin()),
                         std::make_move_iterator(requests->end()));
  }
}

size_t InflightRequests::_doDropCanceled()
{
  size_t removed = 0;
  if (_useMap) {
    for (auto it = _cancelingMap.begin(); it != _cancelingMap.end();) {
      if (it->second && it->second->getStatus() != UCS_INPROGRESS) {
        it = _cancelingMap.erase(it);
        ++removed;
      } else {
        ++it;
      }
    }
  } else {
    auto newEnd = std::remove_if(
      _cancelingVec.begin(), _cancelingVec.end(), [](const std::shared_ptr<Request>& r) {
        return r && r->getStatus() != UCS_INPROGRESS;
      });
    removed = static_cast<size_t>(std::distance(newEnd, _cancelingVec.end()));
    _cancelingVec.erase(newEnd, _cancelingVec.end());
  }
  return removed;
}

size_t InflightRequests::_doCancelingSize() const
{
  return _useMap ? _cancelingMap.size() : _cancelingVec.size();
}

void InflightRequests::_doMergeInflight(std::vector<std::shared_ptr<Request>>* requests)
{
  if (_useMap) {
    for (auto& r : *requests)
      if (r) _inflightMap.emplace(r.get(), std::move(r));
  } else {
    _inflightVec.insert(_inflightVec.end(),
                        std::make_move_iterator(requests->begin()),
                        std::make_move_iterator(requests->end()));
  }
}

void InflightRequests::_doMergeCanceling(std::vector<std::shared_ptr<Request>>* requests)
{
  _doPutCanceling(requests);
}

std::vector<std::shared_ptr<Request>> InflightRequests::_doTakeCanceling()
{
  std::vector<std::shared_ptr<Request>> result;
  if (_useMap) {
    result.reserve(_cancelingMap.size());
    for (auto& kv : _cancelingMap)
      result.push_back(std::move(kv.second));
    _cancelingMap.clear();
  } else {
    result = std::exchange(_cancelingVec, {});
  }
  return result;
}

// ---- Public API -------------------------------------------------------------

InflightRequests::InflightRequests()
{
  const char* env = std::getenv("UCXX_INFLIGHT_REQUESTS_BACKEND");
  if (env != nullptr && std::strcmp(env, "map") == 0) _useMap = true;
}

InflightRequests::~InflightRequests() { cancelAll(); }

size_t InflightRequests::size()
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _doInflightSize();
}

void InflightRequests::insert(const std::shared_ptr<Request>& request)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _doInsert(request);
}

void InflightRequests::remove(const std::shared_ptr<Request>& request)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _doRemove(request);
}

void InflightRequests::merge(TrackedRequests&& trackedRequests)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _doMergeInflight(&trackedRequests.inflight);
  _doMergeCanceling(&trackedRequests.canceling);
}

size_t InflightRequests::cancelAll()
{
  std::vector<std::shared_ptr<Request>> toCancel;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    toCancel = _doTakeInflight();
  }

  size_t total = toCancel.size();
  if (total == 0) return 0;

  ucxx_debug("ucxx::InflightRequests::%s, canceling %lu requests", __func__, total);

  for (auto& r : toCancel) {
    if (r) r->cancel();
  }

  {
    std::lock_guard<std::mutex> lock(_mutex);

    // Keep requests that are still in progress; drop completed ones.
    std::vector<std::shared_ptr<Request>> stillCanceling;
    for (auto& r : toCancel) {
      if (r && r->getStatus() == UCS_INPROGRESS) stillCanceling.push_back(std::move(r));
    }
    _doPutCanceling(&stillCanceling);
  }

  return total;
}

TrackedRequests InflightRequests::release()
{
  std::lock_guard<std::mutex> lock(_mutex);
  TrackedRequests result;
  result.inflight  = _doTakeInflight();
  result.canceling = _doTakeCanceling();
  return result;
}

size_t InflightRequests::getCancelingSize()
{
  std::lock_guard<std::mutex> lock(_mutex);
  _doDropCanceled();
  return _doCancelingSize();
}

}  // namespace ucxx
