/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>
#include <mutex>
#include <utility>

#include <ucxx/delayed_submission.h>
#include <ucxx/log.h>

namespace ucxx {

DelayedSubmission::DelayedSubmission(const bool send,
                                     void* buffer,
                                     const size_t length,
                                     const ucp_tag_t tag)
  : _send(send), _buffer(buffer), _length(length), _tag(tag)
{
}

DelayedSubmissionCallback::DelayedSubmissionCallback(DelayedSubmissionCallbackType callback)
  : _callback(callback)
{
}

DelayedSubmissionCallbackType DelayedSubmissionCallback::get() { return _callback; }

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

    for (auto& dnr : toProcess) {
      auto callback = dnr->get();

      ucxx_trace_req("Submitting request: %p", callback.target<void (*)(std::shared_ptr<void>)>());

      if (callback) callback();
    }
  }
}

void DelayedSubmissionCollection::registerRequest(DelayedSubmissionCallbackType callback)
{
  auto r = std::make_shared<DelayedSubmissionCallback>(callback);

  {
    std::lock_guard<std::mutex> lock(_mutex);
    _collection.push_back(r);
  }
  ucxx_trace_req("Registered submit request: %p",
                 callback.target<void (*)(std::shared_ptr<void>)>());
}

}  // namespace ucxx
