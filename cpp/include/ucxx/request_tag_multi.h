/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <ucxx/endpoint.h>
#include <ucxx/request.h>
#include <ucxx/worker.h>

#include <ucxx/buffer_helper.h>
#include <ucxx/request_helper.h>

#if UCXX_ENABLE_PYTHON
#include <ucxx/python/python_future.h>
#endif

namespace ucxx {

struct UCXXBufferRequest {
  std::shared_ptr<UCXXRequest> request{nullptr};
  std::shared_ptr<std::string> stringBuffer{nullptr};
  std::unique_ptr<UCXXPyBuffer> pyBuffer{nullptr};
};

typedef std::shared_ptr<UCXXBufferRequest> UCXXBufferRequestPtr;

class UCXXRequestTagMulti : public std::enable_shared_from_this<UCXXRequestTagMulti> {
 private:
  std::shared_ptr<UCXXEndpoint> _endpoint = nullptr;
  bool _send{false};
  ucp_tag_t _tag{0};
  size_t _totalFrames{0};
  std::mutex _completedRequestsMutex;
  std::vector<UCXXBufferRequest*> _completedRequests{};
  ucs_status_t _status{UCS_INPROGRESS};
#if UCXX_ENABLE_PYTHON
  std::shared_ptr<PythonFuture> _pythonFuture;
#endif

 public:
  std::vector<UCXXBufferRequestPtr> _bufferRequests{};
  bool _isFilled{false};

 private:
  UCXXRequestTagMulti() = delete;

  // Recv Constructor
  UCXXRequestTagMulti(std::shared_ptr<UCXXEndpoint> endpoint, const ucp_tag_t tag)
    : _endpoint(endpoint), _send(false), _tag(tag)
  {
    ucxx_trace_req("UCXXRequestTagMulti::UCXXRequestTagMulti [recv]: %p, tag: %lx", this, _tag);

    auto worker   = UCXXEndpoint::getWorker(endpoint->getParent());
    _pythonFuture = worker->getPythonFuture();

    callback();
  }

  // Send Constructor
  UCXXRequestTagMulti(std::shared_ptr<UCXXEndpoint> endpoint,
                      std::vector<void*>& buffer,
                      std::vector<size_t>& size,
                      std::vector<int>& isCUDA,
                      const ucp_tag_t tag)
    : _endpoint(endpoint), _send(true), _tag(tag)
  {
    ucxx_trace_req("UCXXRequestTagMulti::UCXXRequestTagMulti [send]: %p, tag: %lx", this, _tag);

    if (size.size() != buffer.size() || isCUDA.size() != buffer.size())
      throw std::runtime_error("All input vectors should be of equal size");

    auto worker   = UCXXEndpoint::getWorker(endpoint->getParent());
    _pythonFuture = worker->getPythonFuture();

    send(buffer, size, isCUDA);
  }

 public:
  friend std::shared_ptr<UCXXRequestTagMulti> tagMultiSend(std::shared_ptr<UCXXEndpoint> endpoint,
                                                           std::vector<void*>& buffer,
                                                           std::vector<size_t>& size,
                                                           std::vector<int>& isCUDA,
                                                           const ucp_tag_t tag)
  {
    ucxx_trace_req("UCXXRequestTagMulti::tagMultiSend");
    return std::shared_ptr<UCXXRequestTagMulti>(
      new UCXXRequestTagMulti(endpoint, buffer, size, isCUDA, tag));
  }

  friend std::shared_ptr<UCXXRequestTagMulti> tagMultiRecv(std::shared_ptr<UCXXEndpoint> endpoint,
                                                           const ucp_tag_t tag)
  {
    ucxx_trace_req("UCXXRequestTagMulti::tagMultiRecv");
    auto ret = std::shared_ptr<UCXXRequestTagMulti>(new UCXXRequestTagMulti(endpoint, tag));
    return ret;
  }

  friend std::vector<std::unique_ptr<UCXXPyBuffer>> tagMultiRecvBlocking(
    std::shared_ptr<UCXXEndpoint> endpoint, ucp_tag_t tag)
  {
    auto worker = UCXXEndpoint::getWorker(endpoint->getParent());

    auto requests = tagMultiRecv(endpoint, tag);

    std::vector<std::shared_ptr<UCXXRequest>> requestsOnly;
    std::vector<std::unique_ptr<UCXXPyBuffer>> recvBuffers;
    for (auto& br : requests->_bufferRequests) {
      requestsOnly.push_back(br->request);
      recvBuffers.push_back(std::move(br->pyBuffer));
    }

    waitRequests(worker, requestsOnly);

    return recvBuffers;
  }

  friend void tagMultiSendBlocking(std::shared_ptr<UCXXEndpoint> endpoint,
                                   std::vector<void*>& buffer,
                                   std::vector<size_t>& size,
                                   std::vector<int>& isCUDA,
                                   ucp_tag_t tag)
  {
    auto worker = UCXXEndpoint::getWorker(endpoint->getParent());

    auto requests = tagMultiSend(endpoint, buffer, size, isCUDA, tag);

    std::vector<std::shared_ptr<UCXXRequest>> requestsOnly;
    for (auto& br : requests->_bufferRequests)
      requestsOnly.push_back(br->request);

    waitRequests(worker, requestsOnly);
  }

  void recvFrames()
  {
    if (_send) throw std::runtime_error("Send requests cannot call recvFrames()");

    std::vector<Header> headers;

    ucxx_trace_req("UCXXRequestTagMulti::recvFrames request: %p, tag: %lx", this, _tag);

    for (auto& br : _bufferRequests)
      headers.push_back(Header(*br->stringBuffer));

    for (auto& h : headers) {
      _totalFrames += h.nframes;
      for (size_t i = 0; i < h.nframes; ++i) {
        auto bufferRequest     = std::make_shared<UCXXBufferRequest>();
        auto buf               = allocateBuffer(h.isCUDA[i], h.size[i]);
        bufferRequest->request = _endpoint->tag_recv(
          buf->data(),
          buf->getSize(),
          _tag,
          std::bind(std::mem_fn(&UCXXRequestTagMulti::markCompleted), this, std::placeholders::_1),
          bufferRequest);
        bufferRequest->pyBuffer = std::move(buf);
        ucxx_trace_req("UCXXRequestTagMulti::recvFrames request: %p, tag: %lx, pyBuffer: %p",
                       this,
                       _tag,
                       bufferRequest->pyBuffer.get());
        _bufferRequests.push_back(bufferRequest);
      }
    }

    _isFilled = true;
    ucxx_trace_req("UCXXRequestTagMulti::recvFrames request: %p, tag: %lx, size: %lu, isFilled: %d",
                   this,
                   _tag,
                   _bufferRequests.size(),
                   _isFilled);
  };

  void markCompleted(std::shared_ptr<void> request)
  {
    std::lock_guard<std::mutex> lock(_completedRequestsMutex);

    /* TODO: Move away from std::shared_ptr<void> to avoid casting void* to
     * UCXXBufferRequest*, or remove pointer holding entirely here since it
     * is not currently used for anything besides counting completed transfers.
     */
    _completedRequests.push_back((UCXXBufferRequest*)request.get());

    if (_completedRequests.size() == _totalFrames) {
      // TODO: Actually handle errors
      _status = UCS_OK;
      _pythonFuture->notify(UCS_OK);
    }
  }

  void recvHeader()
  {
    if (_send) throw std::runtime_error("Send requests cannot call recvFrames()");

    ucxx_trace_req("UCXXRequestTagMulti::recvHeader entering, request: %p, tag: %lx", this, _tag);

    auto bufferRequest          = std::make_shared<UCXXBufferRequest>();
    bufferRequest->stringBuffer = std::make_shared<std::string>(Header::dataSize(), 0);
    bufferRequest->request      = _endpoint->tag_recv(
      bufferRequest->stringBuffer->data(),
      bufferRequest->stringBuffer->size(),
      _tag,
      std::bind(std::mem_fn(&UCXXRequestTagMulti::callback), this, std::placeholders::_1),
      nullptr);

    _bufferRequests.push_back(bufferRequest);
    if (bufferRequest->request->isCompleted()) {
      // TODO: Errors may not be raisable within callback
      bufferRequest->request->checkError();

      // TODO: What if it didn't complete immediately but worker has
      // progressed and completed when it reaches this point? Potential
      // duplication needs to be resolved.
      callback();
    }

    ucxx_trace_req("UCXXRequestTagMulti::recvHeader exiting, request: %p, tag: %lx, empty: %d",
                   this,
                   _tag,
                   _bufferRequests.empty());
  }

  void callback(std::shared_ptr<void> arg = nullptr)
  {
    if (_send) throw std::runtime_error("Send requests cannot call recvFrames()");

    // TODO: Remove arg
    ucxx_trace_req(
      "UCXXRequestTagMulti::callback request: %p, tag: %lx, arg: %p", this, _tag, arg.get());

    if (_bufferRequests.empty()) {
      ucxx_trace_req(
        "UCXXRequestTagMulti::callback first header, request: %p, tag: %lx", this, _tag);
      recvHeader();
    } else {
      const auto& request = _bufferRequests.back();
      auto header         = Header(*_bufferRequests.back()->stringBuffer);

      ucxx_trace_req(
        "UCXXRequestTagMulti::callback request: %p, tag: %lx, "
        "num_requests: %lu, next: %d, request isCompleted: %d, "
        "request status: %s",
        this,
        _tag,
        _bufferRequests.size(),
        header.next,
        request->request->isCompleted(),
        ucs_status_string(request->request->getStatus()));

      if (header.next)
        recvHeader();
      else
        recvFrames();
    }
  }

  void send(std::vector<void*>& buffer, std::vector<size_t>& size, std::vector<int>& isCUDA)
  {
    _totalFrames        = buffer.size();
    size_t totalHeaders = (_totalFrames + HeaderFramesSize - 1) / HeaderFramesSize;

    for (size_t i = 0; i < totalHeaders; ++i) {
      bool hasNext = _totalFrames > (i + 1) * HeaderFramesSize;
      size_t headerFrames =
        hasNext ? HeaderFramesSize : HeaderFramesSize - (HeaderFramesSize * (i + 1) - _totalFrames);

      size_t idx = i * HeaderFramesSize;
      Header header(hasNext, headerFrames, (bool*)&isCUDA[idx], (size_t*)&size[idx]);
      auto serializedHeader = std::make_shared<std::string>(header.serialize());
      auto r = _endpoint->tag_send(serializedHeader->data(), serializedHeader->size(), _tag);

      auto bufferRequest          = std::make_shared<UCXXBufferRequest>();
      bufferRequest->request      = r;
      bufferRequest->stringBuffer = serializedHeader;
      _bufferRequests.push_back(bufferRequest);
    }

    for (size_t i = 0; i < _totalFrames; ++i) {
      auto bufferRequest = std::make_shared<UCXXBufferRequest>();
      auto r             = _endpoint->tag_send(
        buffer[i],
        size[i],
        _tag,
        std::bind(std::mem_fn(&UCXXRequestTagMulti::markCompleted), this, std::placeholders::_1),
        bufferRequest);
      bufferRequest->request = r;
      _bufferRequests.push_back(bufferRequest);
    }

    _isFilled = true;
    ucxx_trace_req("tag_send_multi request: %p, tag: %lx, isFilled: %d", this, _tag, _isFilled);
  }
};

typedef std::shared_ptr<UCXXRequestTagMulti> UCXXRequestTagMultiPtr;

}  // namespace ucxx
