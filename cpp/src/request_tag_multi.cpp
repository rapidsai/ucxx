/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <ucxx/buffer.h>
#include <ucxx/endpoint.h>
#include <ucxx/header.h>
#include <ucxx/request_data.h>
#include <ucxx/request_tag_multi.h>
#include <ucxx/utils/ucx.h>
#include <ucxx/worker.h>

namespace ucxx {

typedef std::pair<Tag, TagMask> TagPair;

BufferRequest::BufferRequest() { ucxx_trace_req("ucxx::BufferRequest created: %p", this); }

BufferRequest::~BufferRequest() { ucxx_trace_req("ucxx::BufferRequest destroyed: %p", this); }

RequestTagMulti::RequestTagMulti(
  std::shared_ptr<Endpoint> endpoint,
  const std::variant<data::TagMultiSend, data::TagMultiReceive> requestData,
  const std::string operationName,
  const bool enablePythonFuture)
  : Request(endpoint, data::getRequestData(requestData), operationName, enablePythonFuture)
{
  auto worker = endpoint->getWorker();
  if (enablePythonFuture) _future = worker->getFuture();
}

RequestTagMulti::~RequestTagMulti()
{
  for (auto& br : _bufferRequests) {
    const auto& ptr = br->request.get();
    if (ptr != nullptr)
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "destroying BufferRequest: %p",
                       br->request.get());

    /**
     * FIXME: The `BufferRequest`s destructor should be doing this, but it seems a
     * reference to the object is lingering and thus it never gets destroyed. This
     * causes a chain effect that prevents `Worker` and `Context` from being destroyed
     * as well. It seems the default destructor fails only for frames, headers seem
     * to be destroyed as expected.
     */
    br->request = nullptr;
  }
}

std::shared_ptr<RequestTagMulti> createRequestTagMulti(
  std::shared_ptr<Endpoint> endpoint,
  const std::variant<data::TagMultiSend, data::TagMultiReceive> requestData,
  const bool enablePythonFuture)
{
  std::shared_ptr<RequestTagMulti> req =
    std::visit(data::dispatch{
                 [&endpoint, &enablePythonFuture](data::TagMultiSend tagMultiSend) {
                   auto req = std::shared_ptr<RequestTagMulti>(new RequestTagMulti(
                     endpoint, tagMultiSend, "tagMultiSend", enablePythonFuture));
                   req->send();
                   return req;
                 },
                 [&endpoint, &enablePythonFuture](data::TagMultiReceive tagMultiReceive) {
                   auto req = std::shared_ptr<RequestTagMulti>(new RequestTagMulti(
                     endpoint, tagMultiReceive, "tagMultiRecv", enablePythonFuture));
                   req->recvCallback(UCS_OK);
                   return req;
                 },
               },
               requestData);

  return req;
}

static TagPair checkAndGetTagPair(const data::RequestData& requestData,
                                  const std::string methodName)
{
  return std::visit(
    data::dispatch{
      [](data::TagMultiReceive tagMultiReceive) {
        return TagPair{tagMultiReceive._tag, tagMultiReceive._tagMask};
      },
      [&methodName](auto) -> TagPair {
        throw std::runtime_error(methodName + "() can only be called by a receive request.");
      },
    },
    requestData);
}

void RequestTagMulti::recvFrames()
{
  auto tagPair = checkAndGetTagPair(_requestData, std::string("recvFrames"));

  std::vector<Header> headers;

  ucxx_trace_req_f(_ownerString.c_str(),
                   this,
                   _request,
                   _operationName.c_str(),
                   "recvFrames, tag: 0x%lx, tagMask: 0x%lx, _bufferRequests.size(): "
                   "%lu",
                   tagPair.first,
                   tagPair.second,
                   _bufferRequests.size());

  for (auto& br : _bufferRequests) {
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "recvFrames, tag: 0x%lx, tagMask: 0x%lx, "
                     "*br->stringBuffer.size(): %lu",
                     tagPair.first,
                     tagPair.second,
                     br->stringBuffer->size());
    headers.push_back(Header(*br->stringBuffer));
  }

  for (auto& h : headers) {
    _totalFrames += h.nframes;
    for (size_t i = 0; i < h.nframes; ++i) {
      auto bufferRequest = std::make_shared<BufferRequest>();
      _bufferRequests.push_back(bufferRequest);
      const auto bufferType  = h.isCUDA[i] ? ucxx::BufferType::RMM : ucxx::BufferType::Host;
      auto buf               = allocateBuffer(bufferType, h.size[i]);
      bufferRequest->request = _endpoint->tagRecv(
        buf->data(),
        buf->getSize(),
        tagPair.first,
        tagPair.second,
        false,
        [this](ucs_status_t status, RequestCallbackUserData arg) {
          return this->markCompleted(status, arg);
        },
        bufferRequest);
      bufferRequest->buffer = buf;
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "recvFrames, tag: 0x%lx, tagMask: 0x%lx, buffer: %p",
                       tagPair.first,
                       tagPair.second,
                       bufferRequest->buffer.get());
    }
  }

  _isFilled = true;
  ucxx_trace_req_f(_ownerString.c_str(),
                   this,
                   _request,
                   _operationName.c_str(),
                   "recvFrames, tag: 0x%lx, tagMask: 0x%lx, size: %lu, isFilled: %d",
                   tagPair.first,
                   tagPair.second,
                   _bufferRequests.size(),
                   _isFilled);
};

void RequestTagMulti::markCompleted(ucs_status_t status, RequestCallbackUserData request)
{
  /**
   * Prevent reference count to self from going to zero and thus cause self to be destroyed
   * while `markCompleted()` executes.
   */
  decltype(shared_from_this()) selfReference = nullptr;
  try {
    selfReference = shared_from_this();
  } catch (std::bad_weak_ptr& exception) {
    ucxx_debug(
      "ucxx::RequestTagMulti: %p destroyed before all markCompleted() callbacks were executed",
      this);
    return;
  }

  TagPair tagPair = std::visit(data::dispatch{
                                 [](data::TagMultiSend tagMultiSend) {
                                   return TagPair{tagMultiSend._tag, TagMaskFull};
                                 },
                                 [](data::TagMultiReceive tagMultiReceive) {
                                   return TagPair{tagMultiReceive._tag, tagMultiReceive._tagMask};
                                 },
                                 [](auto) -> TagPair { throw std::runtime_error("Unreachable"); },
                               },
                               _requestData);

  ucxx_trace_req_f(_ownerString.c_str(),
                   this,
                   _request,
                   _operationName.c_str(),
                   "markCompleted, tag: 0x%lx, tagMask: 0x%lx",
                   tagPair.first,
                   tagPair.second);

  std::lock_guard<std::mutex> lock(_completedRequestsMutex);

  if (_finalStatus == UCS_OK && status != UCS_OK) _finalStatus = status;

  if (++_completedRequests == _totalFrames) {
    setStatus(_finalStatus);

    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "markCompleted, tag: 0x%lx, tagMask: 0x%lx, completed: %lu/%lu, "
                     "final status: %d "
                     "(%s)",
                     tagPair.first,
                     tagPair.second,
                     _completedRequests,
                     _totalFrames,
                     _finalStatus,
                     ucs_status_string(_finalStatus));
  } else {
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "markCompleted, tag: 0x%lx, tagMask: 0x%lx, completed: %lu/%lu",
                     tagPair.first,
                     tagPair.second,
                     _completedRequests,
                     _totalFrames);
  }
}

void RequestTagMulti::recvHeader()
{
  auto tagPair = checkAndGetTagPair(_requestData, std::string("recvHeader"));

  ucxx_trace_req_f(_ownerString.c_str(),
                   this,
                   _request,
                   _operationName.c_str(),
                   "recvHeader entering, tag: 0x%lx, tagMask: 0x%lx",
                   tagPair.first,
                   tagPair.second);

  auto bufferRequest = std::make_shared<BufferRequest>();
  _bufferRequests.push_back(bufferRequest);
  bufferRequest->stringBuffer = std::make_shared<std::string>(Header::dataSize(), 0);
  bufferRequest->request =
    _endpoint->tagRecv(&bufferRequest->stringBuffer->front(),
                       bufferRequest->stringBuffer->size(),
                       tagPair.first,
                       tagPair.second,
                       false,
                       [this](ucs_status_t status, RequestCallbackUserData arg) {
                         return this->recvCallback(status);
                       });

  if (bufferRequest->request->isCompleted()) {
    // TODO: Errors may not be raisable within callback
    bufferRequest->request->checkError();
  }

  ucxx_trace_req_f(_ownerString.c_str(),
                   this,
                   _request,
                   _operationName.c_str(),
                   "recvHeader exiting, tag: 0x%lx, tagMask: 0x%lx, empty: %d",
                   tagPair.first,
                   tagPair.second,
                   _bufferRequests.empty());
}

void RequestTagMulti::recvCallback(ucs_status_t status)
{
  auto tagPair = checkAndGetTagPair(_requestData, std::string("recvCallback"));

  ucxx_trace_req_f(_ownerString.c_str(),
                   this,
                   _request,
                   _operationName.c_str(),
                   "recvCallback, tag: 0x%lx, tagMask: 0x%lx",
                   tagPair.first,
                   tagPair.second);

  if (_bufferRequests.empty()) {
    recvHeader();
  } else {
    if (status == UCS_OK) {
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "recvCallback header received, tag: 0x%lx, tagMask: "
                       "0x%lx",
                       tagPair.first,
                       tagPair.second);
    } else {
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "recvCallback failed receiving header with status %d (%s), "
                       "tag: 0x%lx, "
                       "tagMask: 0x%lx",
                       status,
                       ucs_status_string(status),
                       tagPair.first,
                       tagPair.second);

      _status = status;
      if (_future) _future->notify(status);

      return;
    }

    auto header = Header(*_bufferRequests.back()->stringBuffer);

    if (header.next)
      recvHeader();
    else
      recvFrames();
  }
}

void RequestTagMulti::send()
{
  std::visit(
    data::dispatch{
      [this](data::TagMultiSend tagMultiSend) {
        _totalFrames = tagMultiSend._buffer.size();

        auto headers = Header::buildHeaders(tagMultiSend._length, tagMultiSend._isCUDA);

        for (const auto& header : headers) {
          auto serializedHeader = std::make_shared<std::string>(header.serialize());
          auto bufferRequest    = std::make_shared<BufferRequest>();
          _bufferRequests.push_back(bufferRequest);
          bufferRequest->request = _endpoint->tagSend(
            &serializedHeader->front(), serializedHeader->size(), tagMultiSend._tag, false);
          bufferRequest->stringBuffer = serializedHeader;
        }

        for (size_t i = 0; i < _totalFrames; ++i) {
          auto bufferRequest = std::make_shared<BufferRequest>();
          _bufferRequests.push_back(bufferRequest);
          bufferRequest->request =
            _endpoint->tagSend(tagMultiSend._buffer[i],
                               tagMultiSend._length[i],
                               tagMultiSend._tag,
                               false,
                               [this](ucs_status_t status, RequestCallbackUserData arg) {
                                 return this->markCompleted(status, arg);
                               });
        }

        _isFilled = true;
        ucxx_trace_req_f(_ownerString.c_str(),
                         this,
                         _request,
                         _operationName.c_str(),
                         "send, tag: 0x%lx, isFilled: %d",
                         tagMultiSend._tag,
                         _isFilled);
      },
      [](auto) { throw std::runtime_error("send() can only be called by a sendrequest."); },
    },
    _requestData);
}

void RequestTagMulti::populateDelayedSubmission() {}

void RequestTagMulti::cancel()
{
  for (auto& br : _bufferRequests)
    if (br->request) br->request->cancel();
}

}  // namespace ucxx
