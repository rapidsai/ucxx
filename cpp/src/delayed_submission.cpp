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

RequestDelayedSubmissionCollection::RequestDelayedSubmissionCollection(const std::string name,
                                                                       const bool enabled)
  : BaseDelayedSubmissionCollection<
      std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType>>{name, enabled}
{
}

void RequestDelayedSubmissionCollection::scheduleLog(
  std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item)
{
  ucxx_trace_req("Registered %s: %p", _name.c_str(), item.first.get());
}

void RequestDelayedSubmissionCollection::processItem(
  std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item)
{
  auto& req      = item.first;
  auto& callback = item.second;

  ucxx_trace_req("Submitting %s callbacks: %p", _name.c_str(), req.get());

  if (callback) callback();
}

GenericDelayedSubmissionCollection::GenericDelayedSubmissionCollection(const std::string name)
  : BaseDelayedSubmissionCollection<DelayedSubmissionCallbackType>{name, true}
{
}

void GenericDelayedSubmissionCollection::scheduleLog(DelayedSubmissionCallbackType item)
{
  ucxx_trace_req("Registered %s", _name.c_str());
}

void GenericDelayedSubmissionCollection::processItem(DelayedSubmissionCallbackType callback)
{
  ucxx_trace_req("Submitting %s callback", _name.c_str());

  if (callback) callback();
}

DelayedSubmissionCollection::DelayedSubmissionCollection(bool enableDelayedRequestSubmission)
  : _enableDelayedRequestSubmission(enableDelayedRequestSubmission),
    _requests(RequestDelayedSubmissionCollection{"request", enableDelayedRequestSubmission})
{
}

bool DelayedSubmissionCollection::isDelayedRequestSubmissionEnabled() const
{
  return _enableDelayedRequestSubmission;
}

void DelayedSubmissionCollection::processPre()
{
  _requests.process();

  _genericPre.process();
}

void DelayedSubmissionCollection::processPost() { _genericPost.process(); }

void DelayedSubmissionCollection::registerRequest(std::shared_ptr<Request> request,
                                                  DelayedSubmissionCallbackType callback)
{
  _requests.schedule({request, callback});
}

void DelayedSubmissionCollection::registerGenericPre(DelayedSubmissionCallbackType callback)
{
  _genericPre.schedule(callback);
}

void DelayedSubmissionCollection::registerGenericPost(DelayedSubmissionCallbackType callback)
{
  _genericPost.schedule(callback);
}

}  // namespace ucxx
