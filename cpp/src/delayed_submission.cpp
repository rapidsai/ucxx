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

DelayedSubmissionAm::DelayedSubmissionAm(const ucs_memory_type memoryType) : _memoryType(memoryType)
{
}

DelayedSubmissionTag::DelayedSubmissionTag(const Tag tag, const std::optional<TagMask> tagMask)
  : _tag(tag), _tagMask(tagMask)
{
}

DelayedSubmissionData::DelayedSubmissionData(
  const DelayedSubmissionOperationType operationType,
  const TransferDirection transferDirection,
  const std::variant<std::monostate, DelayedSubmissionAm, DelayedSubmissionTag> data)
  : _operationType(operationType), _transferDirection(transferDirection), _data(data)
{
  if (_operationType == DelayedSubmissionOperationType::Am) {
    if (transferDirection == TransferDirection::Send &&
        !std::holds_alternative<DelayedSubmissionAm>(data))
      throw std::runtime_error(
        "Send Am operations require data to be of type `DelayedSubmissionAm`.");
    if (transferDirection == TransferDirection::Receive &&
        !std::holds_alternative<std::monostate>(data))
      throw std::runtime_error(
        "Receive Am operations do not support data value other than `std::monostate`.");
  } else if (_operationType == DelayedSubmissionOperationType::Tag ||
             _operationType == DelayedSubmissionOperationType::TagMulti) {
    if (!std::holds_alternative<DelayedSubmissionTag>(data))
      throw std::runtime_error(
        "Operations Tag and TagMulti require data to be of type `DelayedSubmissionTag`.");
    if (transferDirection == TransferDirection::Send &&
        std::get<DelayedSubmissionTag>(data)._tagMask)
      throw std::runtime_error("Send Tag and TagMulti operations do not take a tag mask.");
    else if (transferDirection == TransferDirection::Receive &&
             !std::get<DelayedSubmissionTag>(data)._tagMask)
      throw std::runtime_error("Receive Tag and TagMulti operations require a tag mask.");
  } else {
    if (!std::holds_alternative<std::monostate>(data))
      throw std::runtime_error("Type does not support data value other than `std::monostate`.");
  }
}

DelayedSubmissionAm DelayedSubmissionData::getAm() { return std::get<DelayedSubmissionAm>(_data); }

DelayedSubmissionTag DelayedSubmissionData::getTag()
{
  return std::get<DelayedSubmissionTag>(_data);
}

DelayedSubmission::DelayedSubmission(const bool send,
                                     void* buffer,
                                     const size_t length,
                                     const DelayedSubmissionData data)
  : _send(send), _buffer(buffer), _length(length), _data(data)
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
