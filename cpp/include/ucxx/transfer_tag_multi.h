/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>
#include <vector>

#include <ucxx/endpoint.h>
#include <ucxx/request.h>
#include <ucxx/worker.h>

#include <ucxx/buffer_helper.h>
#include <ucxx/request_helper.h>

namespace ucxx {

struct UCXXBufferRequest {
  std::shared_ptr<UCXXRequest> request{nullptr};
  std::shared_ptr<std::string> stringBuffer{nullptr};
  std::unique_ptr<UCXXPyBuffer> pyBuffer{nullptr};
};

typedef std::shared_ptr<UCXXBufferRequest> UCXXBufferRequestPtr;

class UCXXBufferRequests : public std::enable_shared_from_this<UCXXBufferRequests> {
 public:
  std::shared_ptr<UCXXEndpoint> _endpoint = nullptr;
  bool _send{false};
  ucp_tag_t _tag = 0;
  std::vector<UCXXBufferRequestPtr> _bufferRequests{};
  bool _isFilled{false};

 private:
  UCXXBufferRequests() = delete;

  // Recv Constructor
  UCXXBufferRequests(std::shared_ptr<UCXXEndpoint> endpoint, const ucp_tag_t tag)
    : _endpoint(endpoint), _send(false), _tag(tag)
  {
    ucxx_trace_req("UCXXBufferRequests::UCXXBufferRequests [recv]: %p, tag: %lx", this, _tag);
    callback();
  }

  // Send Constructor
  UCXXBufferRequests(std::shared_ptr<UCXXEndpoint> endpoint,
                     std::vector<void*>& buffer,
                     std::vector<size_t>& size,
                     std::vector<int>& isCUDA,
                     const ucp_tag_t tag)
    : _endpoint(endpoint), _send(true), _tag(tag)
  {
    ucxx_trace_req("UCXXBufferRequests::UCXXBufferRequests [send]: %p, tag: %lx", this, _tag);
    if (size.size() != buffer.size() || isCUDA.size() != buffer.size())
      throw std::runtime_error("All input vectors should be of equal size");
    send(buffer, size, isCUDA);
  }

 public:
  friend std::shared_ptr<UCXXBufferRequests> tagMultiSend(std::shared_ptr<UCXXEndpoint> endpoint,
                                                          std::vector<void*>& buffer,
                                                          std::vector<size_t>& size,
                                                          std::vector<int>& isCUDA,
                                                          const ucp_tag_t tag)
  {
    ucxx_trace_req("UCXXBufferRequests::tagMultiSend");
    return std::shared_ptr<UCXXBufferRequests>(
      new UCXXBufferRequests(endpoint, buffer, size, isCUDA, tag));
  }

  friend std::shared_ptr<UCXXBufferRequests> tagMultiRecv(std::shared_ptr<UCXXEndpoint> endpoint,
                                                          const ucp_tag_t tag)
  {
    ucxx_trace_req("UCXXBufferRequests::tagMultiRecv");
    auto ret = std::shared_ptr<UCXXBufferRequests>(new UCXXBufferRequests(endpoint, tag));
    return ret;
  }

  void recvFrames()
  {
    if (_send) throw std::runtime_error("Send requests cannot call recvFrames()");

    std::vector<Header> headers;

    ucxx_trace_req("UCXXBufferRequests::recvFrames request: %p, tag: %lx", this, _tag);

    for (auto& br : _bufferRequests)
      headers.push_back(Header(*br->stringBuffer));

    for (auto& h : headers) {
      for (size_t i = 0; i < h.nframes; ++i) {
        auto bufferRequest      = std::make_shared<UCXXBufferRequest>();
        auto buf                = allocateBuffer(h.isCUDA[i], h.size[i]);
        bufferRequest->request  = _endpoint->tag_recv(buf->data(), buf->getSize(), _tag);
        bufferRequest->pyBuffer = std::move(buf);
        ucxx_trace_req("UCXXBufferRequests::recvFrames request: %p, tag: %lx, pyBuffer: %p",
                       this,
                       _tag,
                       bufferRequest->pyBuffer.get());
        _bufferRequests.push_back(bufferRequest);
      }
    }

    _isFilled = true;
    ucxx_trace_req("UCXXBufferRequests::recvFrames request: %p, tag: %lx, size: %lu, isFilled: %d",
                   this,
                   _tag,
                   _bufferRequests.size(),
                   _isFilled);
  };

  void recvHeader()
  {
    if (_send) throw std::runtime_error("Send requests cannot call recvFrames()");

    ucxx_trace_req("UCXXBufferRequests::recvHeader entering, request: %p, tag: %lx", this, _tag);

    auto bufferRequest          = std::make_shared<UCXXBufferRequest>();
    bufferRequest->stringBuffer = std::make_shared<std::string>(Header::dataSize(), 0);
    bufferRequest->request      = _endpoint->tag_recv(
      bufferRequest->stringBuffer->data(),
      bufferRequest->stringBuffer->size(),
      _tag,
      std::bind(std::mem_fn(&UCXXBufferRequests::callback), this, std::placeholders::_1),
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

    ucxx_trace_req("UCXXBufferRequests::recvHeader exiting, request: %p, tag: %lx, empty: %d",
                   this,
                   _tag,
                   _bufferRequests.empty());
  }

  void callback(std::shared_ptr<void> arg = nullptr)
  {
    if (_send) throw std::runtime_error("Send requests cannot call recvFrames()");

    // TODO: Remove arg
    ucxx_trace_req(
      "UCXXBufferRequests::callback request: %p, tag: %lx, arg: %p", this, _tag, arg.get());

    if (_bufferRequests.empty()) {
      ucxx_trace_req(
        "UCXXBufferRequests::callback first header, request: %p, tag: %lx", this, _tag);
      recvHeader();
    } else {
      const auto& request = _bufferRequests.back();
      auto header         = Header(*_bufferRequests.back()->stringBuffer);

      ucxx_trace_req(
        "UCXXBufferRequests::callback request: %p, tag: %lx, "
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
    size_t totalFrames  = buffer.size();
    size_t totalHeaders = (totalFrames + HeaderFramesSize - 1) / HeaderFramesSize;

    for (size_t i = 0; i < totalHeaders; ++i) {
      bool hasNext = totalFrames > (i + 1) * HeaderFramesSize;
      size_t headerFrames =
        hasNext ? HeaderFramesSize : HeaderFramesSize - (HeaderFramesSize * (i + 1) - totalFrames);

      size_t idx = i * HeaderFramesSize;
      Header header(hasNext, headerFrames, (bool*)&isCUDA[idx], (size_t*)&size[idx]);
      auto serializedHeader = std::make_shared<std::string>(header.serialize());
      auto r = _endpoint->tag_send(serializedHeader->data(), serializedHeader->size(), _tag);

      auto bufferRequest          = std::make_shared<UCXXBufferRequest>();
      bufferRequest->request      = r;
      bufferRequest->stringBuffer = serializedHeader;
      _bufferRequests.push_back(bufferRequest);
    }

    for (size_t i = 0; i < totalFrames; ++i) {
      auto r                 = _endpoint->tag_send(buffer[i], size[i], _tag);
      auto bufferRequest     = std::make_shared<UCXXBufferRequest>();
      bufferRequest->request = r;
      _bufferRequests.push_back(bufferRequest);
    }

    _isFilled = true;
    ucxx_trace_req("tag_send_multi request: %p, tag: %lx, isFilled: %d", this, _tag, _isFilled);
  }
};

typedef std::shared_ptr<UCXXBufferRequests> UCXXBufferRequestsPtr;

std::vector<std::unique_ptr<UCXXPyBuffer>> tag_recv_multi_b(std::shared_ptr<UCXXEndpoint> endpoint,
                                                            ucp_tag_t tag)
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

void tag_send_multi_b(std::shared_ptr<UCXXEndpoint> endpoint,
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

}  // namespace ucxx