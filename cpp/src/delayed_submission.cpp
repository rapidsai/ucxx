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
  decltype(_requests) requestsToProcess;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    // Move _requests to a local copy in order to to hold the lock for as
    // short as possible
    requestsToProcess = std::move(_requests);
  }
  if (requestsToProcess.size() > 0) {
    ucxx_trace_req("Submitting %lu requests", requestsToProcess.size());
    for (auto& pair : requestsToProcess) {
      auto& req      = pair.first;
      auto& callback = pair.second;

      ucxx_trace_req("Submitting request: %p", req.get());

      if (callback) callback();
    }
  }
  decltype(_genericPre) callbacks;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    // Move _genericPre to a local copy in order to to hold the lock for as
    // short as possible
    callbacks = std::move(_genericPre);
  }

  if (callbacks.size() > 0) {
    ucxx_trace_req("Submitting %lu generic", callbacks.size());

    for (auto& callback : callbacks) {
      ucxx_trace_req("Submitting generic");

      if (callback) callback();
    }
  }
}

void DelayedSubmissionCollection::processPost()
{
  decltype(_genericPost) callbacks;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    // Move _genericPost to a local copy in order to to hold the lock for as
    // short as possible
    callbacks = std::move(_genericPost);
  }

  if (callbacks.size() > 0) {
    ucxx_trace_req("Submitting %lu generic", callbacks.size());

    for (auto& callback : callbacks) {
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
