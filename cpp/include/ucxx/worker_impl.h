/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

namespace ucxx {

inline size_t UCXXWorker::cancelInflightRequests()
{
  // Fast path when no requests have been scheduled for cancelation
  if (_inflightRequestsToCancel->size() == 0) return 0;

  size_t total = 0;
  std::lock_guard<std::mutex> lock(_inflightMutex);

  for (auto& r : *_inflightRequestsToCancel) {
    if (auto request = r.second.lock()) {
      request->cancel();
      ++total;
    }
  }

  _inflightRequestsToCancel->clear();
  return total;
}

}  // namespace ucxx
