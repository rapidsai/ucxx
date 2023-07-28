/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <ucxx/buffer.h>
#include <ucxx/endpoint.h>
#include <ucxx/header.h>
#include <ucxx/request_tag_multi.h>
#include <ucxx/utils/ucx.h>
#include <ucxx/worker.h>

namespace ucxx {

RequestTagMulti::RequestTagMulti(std::shared_ptr<Endpoint> endpoint,
                                 const bool send,
                                 const ucp_tag_t tag,
                                 const bool enablePythonFuture)
  : Request(endpoint,
            std::make_shared<DelayedSubmission>(!send, nullptr, 0, 0),
            std::string(send ? "tagMultiSend" : "tagMultiRecv"),
            enablePythonFuture),
    _send(send),
    _tag(tag)
{
  ucxx_trace_req("RequestTagMulti::RequestTagMulti: %p, send: %d, tag: %lx", this, send, _tag);

  auto worker = endpoint->getWorker();
  if (enablePythonFuture) _future = worker->getFuture();
}

RequestTagMulti::~RequestTagMulti()
{
  for (auto& br : _bufferRequests) {
    const auto& ptr = br->request.get();
    if (ptr != nullptr)
      ucxx_trace("RequestTagMulti destroying BufferRequest: %p", br->request.get());

    /**
     * FIXME: The `BufferRequest`s destructor should be doing this, but it seems a
     * reference to the object is lingering and thus it never gets destroyed. This
     * causes a chain effect that prevents `Worker` and `Context` from being destroyed
     * as well. It seems the default destructor fails only for frames, headers seem
     * to be destroyed as expected.
     */
    br->request = nullptr;
  }
  ucxx_trace("RequestTagMulti destroyed: %p", this);
}

std::shared_ptr<RequestTagMulti> createRequestTagMultiSend(std::shared_ptr<Endpoint> endpoint,
                                                           const std::vector<void*>& buffer,
                                                           const std::vector<size_t>& size,
                                                           const std::vector<int>& isCUDA,
                                                           const ucp_tag_t tag,
                                                           const bool enablePythonFuture)
{
  ucxx_trace_req("RequestTagMulti::tagMultiSend");
  auto ret =
    std::shared_ptr<RequestTagMulti>(new RequestTagMulti(endpoint, true, tag, enablePythonFuture));

  if (size.size() != buffer.size() || isCUDA.size() != buffer.size())
    throw std::runtime_error("All input vectors should be of equal size");

  ucxx_trace("RequestTagMulti created: %p", ret.get());

  ret->send(buffer, size, isCUDA);

  return ret;
}

std::shared_ptr<RequestTagMulti> createRequestTagMultiRecv(std::shared_ptr<Endpoint> endpoint,
                                                           const ucp_tag_t tag,
                                                           const bool enablePythonFuture)
{
  ucxx_trace_req("RequestTagMulti::tagMultiRecv");
  auto ret =
    std::shared_ptr<RequestTagMulti>(new RequestTagMulti(endpoint, false, tag, enablePythonFuture));

  ucxx_trace("RequestTagMulti created: %p", ret.get());

  ret->recvCallback(UCS_OK);

  return ret;
}

void RequestTagMulti::recvFrames()
{
  if (_send) throw std::runtime_error("Send requests cannot call recvFrames()");

  std::vector<Header> headers;

  ucxx_trace_req("RequestTagMulti::recvFrames request: %p, tag: %lx, _bufferRequests.size(): %lu",
                 this,
                 _tag,
                 _bufferRequests.size());

  for (auto& br : _bufferRequests) {
    ucxx_trace_req(
      "RequestTagMulti::recvFrames request: %p, tag: %lx, *br->stringBuffer.size(): %lu",
      this,
      _tag,
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
        _tag,
        false,
        [this](ucs_status_t status, RequestCallbackUserData arg) {
          return this->markCompleted(status, arg);
        },
        bufferRequest);
      bufferRequest->buffer = buf;
      ucxx_trace_req("RequestTagMulti::recvFrames request: %p, tag: %lx, buffer: %p",
                     this,
                     _tag,
                     bufferRequest->buffer);
    }
  }

  _isFilled = true;
  ucxx_trace_req("RequestTagMulti::recvFrames request: %p, tag: %lx, size: %lu, isFilled: %d",
                 this,
                 _tag,
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
    ucxx_debug("RequestTagMulti %p destroyed before all markCompleted() callbacks were executed",
               this);
    return;
  }

  ucxx_trace_req("RequestTagMulti::markCompleted request: %p, tag: %lx", this, _tag);
  std::lock_guard<std::mutex> lock(_completedRequestsMutex);

  /* TODO: Move away from std::shared_ptr<void> to avoid casting void* to
   * BufferRequest*, or remove pointer holding entirely here since it
   * is not currently used for anything besides counting completed transfers.
   */
  ++_completedRequests;

  if (_completedRequests == _totalFrames) {
    auto s = UCS_OK;

    // Get the first non-UCS_OK status and set that as complete status
    for (const auto& br : _bufferRequests) {
      if (br->request) {
        s = br->request->getStatus();
        if (s != UCS_OK) break;
      }
    }

    setStatus(s);
  }

  ucxx_trace_req("RequestTagMulti::markCompleted request: %p, tag: %lx, completed: %lu/%lu",
                 this,
                 _tag,
                 _completedRequests,
                 _totalFrames);
}

void RequestTagMulti::recvHeader()
{
  if (_send) throw std::runtime_error("Send requests cannot call recvHeader()");

  ucxx_trace_req("RequestTagMulti::recvHeader entering, request: %p, tag: %lx", this, _tag);

  auto bufferRequest = std::make_shared<BufferRequest>();
  _bufferRequests.push_back(bufferRequest);
  bufferRequest->stringBuffer = std::make_shared<std::string>(Header::dataSize(), 0);
  bufferRequest->request      = _endpoint->tagRecv(
    &bufferRequest->stringBuffer->front(),
    bufferRequest->stringBuffer->size(),
    _tag,
    false,
    [this](ucs_status_t status, RequestCallbackUserData arg) { return this->recvCallback(status); },
    nullptr);

  if (bufferRequest->request->isCompleted()) {
    // TODO: Errors may not be raisable within callback
    bufferRequest->request->checkError();
  }

  ucxx_trace_req("RequestTagMulti::recvHeader exiting, request: %p, tag: %lx, empty: %d",
                 this,
                 _tag,
                 _bufferRequests.empty());
}

void RequestTagMulti::recvCallback(ucs_status_t status)
{
  if (_send) throw std::runtime_error("Send requests cannot call recvCallback()");

  ucxx_trace_req("RequestTagMulti::recvCallback request: %p, tag: %lx", this, _tag);

  if (_bufferRequests.empty()) {
    recvHeader();
  } else {
    const auto request = _bufferRequests.back();

    if (status == UCS_OK) {
      ucxx_trace_req(
        "RequestTagMulti::recvCallback header received, multi request: %p, tag: %lx", this, _tag);
    } else {
      ucxx_trace_req(
        "RequestTagMulti::recvCallback failed receiving header with status %d (%s), multi request: "
        "%p, "
        "tag: %lx",
        status,
        ucs_status_string(status),
        this,
        _tag);

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

void RequestTagMulti::send(const std::vector<void*>& buffer,
                           const std::vector<size_t>& size,
                           const std::vector<int>& isCUDA)
{
  _totalFrames = buffer.size();

  if ((size.size() != _totalFrames) || (isCUDA.size() != _totalFrames))
    throw std::length_error("buffer, size and isCUDA must have the same length");

  auto headers = Header::buildHeaders(size, isCUDA);

  for (const auto& header : headers) {
    auto serializedHeader = std::make_shared<std::string>(header.serialize());
    auto r = _endpoint->tagSend(&serializedHeader->front(), serializedHeader->size(), _tag, false);

    auto bufferRequest          = std::make_shared<BufferRequest>();
    bufferRequest->request      = r;
    bufferRequest->stringBuffer = serializedHeader;
    _bufferRequests.push_back(bufferRequest);
  }

  for (size_t i = 0; i < _totalFrames; ++i) {
    auto bufferRequest = std::make_shared<BufferRequest>();
    auto r             = _endpoint->tagSend(
      buffer[i],
      size[i],
      _tag,
      false,
      [this](ucs_status_t status, RequestCallbackUserData arg) {
        return this->markCompleted(status, arg);
      },
      bufferRequest);
    bufferRequest->request = r;
    _bufferRequests.push_back(bufferRequest);
  }

  _isFilled = true;
  ucxx_trace_req(
    "RequestTagMulti::send request: %p, tag: %lx, isFilled: %d", this, _tag, _isFilled);
}

void RequestTagMulti::populateDelayedSubmission() {}

void RequestTagMulti::cancel()
{
  for (auto& br : _bufferRequests)
    if (br->request) br->request->cancel();
}

}  // namespace ucxx
