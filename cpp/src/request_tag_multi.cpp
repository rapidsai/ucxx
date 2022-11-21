/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <ucxx/buffer.h>
#include <ucxx/endpoint.h>
#include <ucxx/header.h>
#include <ucxx/request.h>
#include <ucxx/request_helper.h>
#include <ucxx/request_tag_multi.h>
#include <ucxx/worker.h>

#if UCXX_ENABLE_PYTHON
#include <ucxx/python/python_future.h>
#else
typedef void PyObject;
#endif

namespace ucxx {

RequestTagMulti::RequestTagMulti(std::shared_ptr<Endpoint> endpoint,
                                 const ucp_tag_t tag,
                                 const bool enablePythonFuture)
  : _endpoint(endpoint), _send(false), _tag(tag)
{
  ucxx_trace_req("RequestTagMulti::RequestTagMulti [recv]: %p, tag: %lx", this, _tag);

  auto worker = Endpoint::getWorker(endpoint->getParent());
#if UCXX_ENABLE_PYTHON
  if (enablePythonFuture) _pythonFuture = worker->getPythonFuture();
#endif

  callback();
}

RequestTagMulti::RequestTagMulti(std::shared_ptr<Endpoint> endpoint,
                                 std::vector<void*>& buffer,
                                 std::vector<size_t>& size,
                                 std::vector<int>& isCUDA,
                                 const ucp_tag_t tag,
                                 const bool enablePythonFuture)
  : _endpoint(endpoint), _send(true), _tag(tag)
{
  ucxx_trace_req("RequestTagMulti::RequestTagMulti [send]: %p, tag: %lx", this, _tag);

  if (size.size() != buffer.size() || isCUDA.size() != buffer.size())
    throw std::runtime_error("All input vectors should be of equal size");

  auto worker = Endpoint::getWorker(endpoint->getParent());
#if UCXX_ENABLE_PYTHON
  if (enablePythonFuture) _pythonFuture = worker->getPythonFuture();
#endif

  send(buffer, size, isCUDA);
}

std::shared_ptr<RequestTagMulti> createRequestTagMultiSend(std::shared_ptr<Endpoint> endpoint,
                                                           std::vector<void*>& buffer,
                                                           std::vector<size_t>& size,
                                                           std::vector<int>& isCUDA,
                                                           const ucp_tag_t tag,
                                                           const bool enablePythonFuture)
{
  ucxx_trace_req("RequestTagMulti::tagMultiSend");
  return std::shared_ptr<RequestTagMulti>(
    new RequestTagMulti(endpoint, buffer, size, isCUDA, tag, enablePythonFuture));
}

std::shared_ptr<RequestTagMulti> createRequestTagMultiRecv(std::shared_ptr<Endpoint> endpoint,
                                                           const ucp_tag_t tag,
                                                           const bool enablePythonFuture)
{
  ucxx_trace_req("RequestTagMulti::tagMultiRecv");
  auto ret =
    std::shared_ptr<RequestTagMulti>(new RequestTagMulti(endpoint, tag, enablePythonFuture));
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
        std::bind(std::mem_fn(&RequestTagMulti::markCompleted), this, std::placeholders::_1),
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

void RequestTagMulti::markCompleted(std::shared_ptr<void> request)
{
  ucxx_trace_req("RequestTagMulti::markCompleted request: %p, tag: %lx", this, _tag);
  std::lock_guard<std::mutex> lock(_completedRequestsMutex);

  /* TODO: Move away from std::shared_ptr<void> to avoid casting void* to
   * BufferRequest*, or remove pointer holding entirely here since it
   * is not currently used for anything besides counting completed transfers.
   */
  _completedRequests.push_back((BufferRequest*)request.get());

  if (_completedRequests.size() == _totalFrames) {
    // TODO: Actually handle errors
    _status = UCS_OK;
#if UCXX_ENABLE_PYTHON
    if (_pythonFuture) _pythonFuture->notify(UCS_OK);
#endif
  }

  ucxx_trace_req("RequestTagMulti::markCompleted request: %p, tag: %lx, completed: %lu/%lu",
                 this,
                 _tag,
                 _completedRequests.size(),
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
    std::bind(std::mem_fn(&RequestTagMulti::callback), this, std::placeholders::_1),
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

void RequestTagMulti::callback(std::shared_ptr<void> arg)
{
  if (_send) throw std::runtime_error("Send requests cannot call callback()");

  // TODO: Remove arg
  ucxx_trace_req("RequestTagMulti::callback request: %p, tag: %lx, arg: %p", this, _tag, arg.get());

  if (_bufferRequests.empty()) {
    ucxx_trace_req("RequestTagMulti::callback first header, request: %p, tag: %lx", this, _tag);
    recvHeader();
  } else {
    const auto request = _bufferRequests.back();
    auto header        = Header(*_bufferRequests.back()->stringBuffer);

    // FIXME: request->request is not available when recvHeader completes immediately,
    // the `tagRecv` operation hasn't returned yet.
    // ucxx_trace_req(
    //   "RequestTagMulti::callback request: %p, tag: %lx, "
    //   "num_requests: %lu, next: %d, request isCompleted: %d, "
    //   "request status: %s",
    //   this,
    //   _tag,
    //   _bufferRequests.size(),
    //   header.next,
    //   request->request->isCompleted(),
    //   ucs_status_string(request->request->getStatus()));

    if (header.next)
      recvHeader();
    else
      recvFrames();
  }
}

void RequestTagMulti::send(std::vector<void*>& buffer,
                           std::vector<size_t>& size,
                           std::vector<int>& isCUDA)
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
      std::bind(std::mem_fn(&RequestTagMulti::markCompleted), this, std::placeholders::_1),
      bufferRequest);
    bufferRequest->request = r;
    _bufferRequests.push_back(bufferRequest);
  }

  _isFilled = true;
  ucxx_trace_req(
    "RequestTagMulti::send request: %p, tag: %lx, isFilled: %d", this, _tag, _isFilled);
}

ucs_status_t RequestTagMulti::getStatus() { return _status; }

PyObject* RequestTagMulti::getPyFuture()
{
#if UCXX_ENABLE_PYTHON
  if (_pythonFuture)
    return (PyObject*)_pythonFuture->getHandle();
  else
#endif
    return nullptr;
}

void RequestTagMulti::checkError()
{
  switch (_status) {
    case UCS_OK:
    case UCS_INPROGRESS: return;
    case UCS_ERR_CANCELED: throw CanceledError(ucs_status_string(_status)); break;
    default: throw Error(ucs_status_string(_status)); break;
  }
}

template <typename Rep, typename Period>
bool RequestTagMulti::isCompleted(std::chrono::duration<Rep, Period> period)
{
  return _status != UCS_INPROGRESS;
}

bool RequestTagMulti::isCompleted(int64_t periodNs)
{
  return isCompleted(std::chrono::nanoseconds(periodNs));
}

}  // namespace ucxx
