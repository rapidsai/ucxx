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

RequestDelayedSubmissionCollection::RequestDelayedSubmissionCollection(const std::string_view name)
  : BaseDelayedSubmissionCollection<
      std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType>>{name}
{
}

void RequestDelayedSubmissionCollection::scheduleLog(
  std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item)
{
  ucxx_trace_req("Registered %s: %p", _name, item.first.get());
}

void RequestDelayedSubmissionCollection::processItem(
  std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item)
{
  auto& req      = item.first;
  auto& callback = item.second;

  ucxx_trace_req("Submitting %s callbacks: %p", _name, req.get());

  if (callback) callback();
}

GenericDelayedSubmissionCollection::GenericDelayedSubmissionCollection(const std::string_view name)
  : BaseDelayedSubmissionCollection<DelayedSubmissionCallbackType>{name}
{
}

void GenericDelayedSubmissionCollection::scheduleLog(DelayedSubmissionCallbackType item)
{
  ucxx_trace_req("Registered %s", _name);
}

void GenericDelayedSubmissionCollection::processItem(DelayedSubmissionCallbackType callback)
{
  ucxx_trace_req("Submitting %s callback", _name);

  if (callback) callback();
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
  _requests.process();

  _genericPre.process();
}

void DelayedSubmissionCollection::processPost() { _genericPost.process(); }

void DelayedSubmissionCollection::registerRequest(std::shared_ptr<Request> request,
                                                  DelayedSubmissionCallbackType callback)
{
  _requests.schedule({request, callback}, isDelayedRequestSubmissionEnabled());
}

void DelayedSubmissionCollection::registerGenericPre(DelayedSubmissionCallbackType callback)
{
  _genericPre.schedule({callback}, true);
}

void DelayedSubmissionCollection::registerGenericPost(DelayedSubmissionCallbackType callback)
{
  _genericPost.schedule({callback}, true);
}

}  // namespace ucxx
