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

RequestDelayedSubmissionCollection::RequestDelayedSubmissionCollection(const std::string name,
                                                                       const bool enabled)
  : BaseDelayedSubmissionCollection<
      std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType>>{name, enabled}
{
}

void RequestDelayedSubmissionCollection::scheduleLog(
  uint64_t id, std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item)
{
  ucxx_trace_req("Registered %s [%lu]: %p", _name.c_str(), id, item.first.get());
}

void RequestDelayedSubmissionCollection::processItem(
  uint64_t id, std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item)
{
  auto& req      = item.first;
  auto& callback = item.second;

  ucxx_trace_req("Submitting %s [%lu] callback: %p", _name.c_str(), id, req.get());

  if (callback) callback();
}

GenericDelayedSubmissionCollection::GenericDelayedSubmissionCollection(const std::string name)
  : BaseDelayedSubmissionCollection<DelayedSubmissionCallbackType>{name, true}
{
}

void GenericDelayedSubmissionCollection::scheduleLog(uint64_t id,
                                                     DelayedSubmissionCallbackType item)
{
  ucxx_trace_req("Registered %s [%lu]", _name.c_str(), id);
}

void GenericDelayedSubmissionCollection::processItem(uint64_t id,
                                                     DelayedSubmissionCallbackType callback)
{
  ucxx_trace_req("Submitting %s [%lu] callback", _name.c_str(), id);

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

uint64_t DelayedSubmissionCollection::registerGenericPre(DelayedSubmissionCallbackType callback)
{
  return _genericPre.schedule(callback);
}

uint64_t DelayedSubmissionCollection::registerGenericPost(DelayedSubmissionCallbackType callback)
{
  return _genericPost.schedule(callback);
}

void DelayedSubmissionCollection::cancelGenericPre(uint64_t id) { _genericPre.cancel(id); }

void DelayedSubmissionCollection::cancelGenericPost(uint64_t id) { _genericPre.cancel(id); }

}  // namespace ucxx
