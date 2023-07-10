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

DelayedSubmissionCollection::DelayedSubmissionCollection(bool enableDelayedRequestSubmission)
  : _enableDelayedRequestSubmission(enableDelayedRequestSubmission)
{
}

bool DelayedSubmissionCollection::isDelayedRequestSubmissionEnabled() const
{
  return _enableDelayedRequestSubmission;
}

void DelayedSubmissionCollection::processPre()
{
  if (_requests.size() > 0) {
    ucxx_trace_req("Submitting %lu requests", _requests.size());

    // Move _requests to a local copy in order to to hold the lock for as
    // short as possible
    decltype(_requests) toProcess;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      toProcess = std::move(_requests);
    }

    for (auto& pair : toProcess) {
      auto& req      = pair.first;
      auto& callback = pair.second;

      ucxx_trace_req("Submitting request: %p", req.get());

      if (callback) callback();
    }
  }

  if (_genericPre.size() > 0) {
    ucxx_trace_req("Submitting %lu generic", _genericPre.size());

    // Move _genericPre to a local copy in order to to hold the lock for as
    // short as possible
    decltype(_genericPre) toProcess;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      toProcess = std::move(_genericPre);
    }

    for (auto& callback : toProcess) {
      ucxx_trace_req("Submitting generic");

      if (callback) callback();
    }
  }
}

void DelayedSubmissionCollection::processPost()
{
  if (_genericPost.size() > 0) {
    ucxx_trace_req("Submitting %lu generic", _genericPost.size());

    // Move _genericPost to a local copy in order to to hold the lock for as
    // short as possible
    decltype(_genericPost) toProcess;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      toProcess = std::move(_genericPost);
    }

    for (auto& callback : toProcess) {
      ucxx_trace_req("Submitting generic");

      if (callback) callback();
    }
  }
}

void DelayedSubmissionCollection::registerRequest(std::shared_ptr<Request> request,
                                                  DelayedSubmissionCallbackType callback)
{
  if (!isDelayedRequestSubmissionEnabled()) throw std::runtime_error("Context not initialized");

  {
    std::lock_guard<std::mutex> lock(_mutex);
    _requests.push_back({request, callback});
  }
  ucxx_trace_req("Registered submit request: %p", request.get());
}

void DelayedSubmissionCollection::registerGenericPre(DelayedSubmissionCallbackType callback)
{
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _genericPre.push_back({callback});
  }
  ucxx_trace_req("Registered generic");
}

void DelayedSubmissionCollection::registerGenericPost(DelayedSubmissionCallbackType callback)
{
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _genericPost.push_back({callback});
  }
  ucxx_trace_req("Registered generic");
}

}  // namespace ucxx
