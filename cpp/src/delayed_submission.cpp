/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <mutex>
#include <utility>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/log.h>

namespace ucxx {

DelayedSubmission::DelayedSubmission(const bool send,
                                     void* buffer,
                                     const size_t length,
                                     const ucp_tag_t tag,
                                     const ucs_memory_type_t memoryType)
  : _send(send), _buffer(buffer), _length(length), _tag(tag), _memoryType(memoryType)
{
}

void DelayedSubmissionCollection::process()
{
  if (_collection.size() > 0) {
    ucxx_trace_req("Submitting %lu requests", _collection.size());

    // Move _collection to a local copy in order to to hold the lock for as
    // short as possible
    decltype(_collection) toProcess;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      toProcess = std::move(_collection);
    }

    for (auto& pair : toProcess) {
      auto& req      = pair.first;
      auto& callback = pair.second;

      ucxx_trace_req("Submitting request: %p", req.get());

      if (callback) callback();
    }
  }
}

void DelayedSubmissionCollection::registerRequest(std::shared_ptr<Request> request,
                                                  DelayedSubmissionCallbackType callback)
{
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _collection.push_back({request, callback});
  }
  ucxx_trace_req("Registered submit request: %p", request.get());
}

}  // namespace ucxx
