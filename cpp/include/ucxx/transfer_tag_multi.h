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

namespace ucxx
{

struct UCXXBufferRequest
{
    std::shared_ptr<UCXXRequest> request{nullptr};
    std::shared_ptr<std::string> stringBuffer{nullptr};
    std::unique_ptr<UCXXPyBuffer> pyBuffer{nullptr};
};

typedef std::shared_ptr<UCXXBufferRequest> UCXXBufferRequestPtr;

struct UCXXBufferRequests
{
    std::vector<UCXXBufferRequestPtr> bufferRequests{};
    bool isFilled{false};
    std::shared_ptr<UCXXEndpoint> endpoint = nullptr;
    ucp_tag_t tag = 0;
};

typedef std::shared_ptr<UCXXBufferRequests> UCXXBufferRequestsPtr;

void tag_recv_multi_callback(std::shared_ptr<void> ucxxBufferRequests);

void tag_recv_multi_frames(std::shared_ptr<void> ucxxBufferRequests)
{
    auto bufferRequests = std::reinterpret_pointer_cast<UCXXBufferRequests>(ucxxBufferRequests);
    auto endpoint = bufferRequests->endpoint;
    auto tag = bufferRequests->tag;
    std::vector<Header> headers;

    ucxx_trace_req("tag_recv_multi_frames request: %p, tag: %lx",
                   bufferRequests.get(), tag);

    for (auto& br : bufferRequests->bufferRequests)
        // headers.push_back(Header(*br->bufferRequests->stringBuffer));
        headers.push_back(Header(*br->stringBuffer));

    for (auto& h : headers)
    {
        for (size_t i = 0; i < h.nframes; ++i)
        {
            auto bufferRequest = std::make_shared<UCXXBufferRequest>();
            auto buf = allocateBuffer(h.isCUDA[i], h.size[i]);
            bufferRequest->request = endpoint->tag_recv(buf->data(), buf->getSize(), tag);
            bufferRequest->pyBuffer = std::move(buf);
            ucxx_trace_req("tag_recv_multi_frames request: %p, tag: %lx, pyBuffer: %p",
                           bufferRequests.get(), tag, bufferRequest->pyBuffer.get());
            bufferRequests->bufferRequests.push_back(bufferRequest);
        }
    }

    bufferRequests->isFilled = true;
    ucxx_trace_req("tag_recv_multi_frames request: %p, tag: %lx, size: %lu, isFilled: %d",
                   bufferRequests.get(), tag, bufferRequests->bufferRequests.size(),
                   bufferRequests->isFilled);
};

void tag_recv_multi_header(std::shared_ptr<void> ucxxBufferRequests)
{
    auto bufferRequests = std::reinterpret_pointer_cast<UCXXBufferRequests>(ucxxBufferRequests);
    auto endpoint = bufferRequests->endpoint;
    auto tag = bufferRequests->tag;

    ucxx_trace_req("tag_recv_multi_header entering, request: %p, tag: %lx",
                   bufferRequests.get(), tag);

    auto bufferRequest = std::make_shared<UCXXBufferRequest>();
    bufferRequest->stringBuffer = std::make_shared<std::string>(Header::dataSize(), 0);
    bufferRequest->request = endpoint->tag_recv(bufferRequest->stringBuffer->data(),
                                                bufferRequest->stringBuffer->size(),
                                                tag,
                                                (void*)tag_recv_multi_callback,
                                                ucxxBufferRequests);

    bufferRequests->bufferRequests.push_back(bufferRequest);
    if (bufferRequest->request->isCompleted())
    {
        // TODO: Errors may not be raisable within callback
        bufferRequest->request->checkError();

        // TODO: What if it didn't complete immediately but worker has
        // progressed and completed when it reaches this point? Potential
        // duplication needs to be resolved.
        tag_recv_multi_callback(ucxxBufferRequests);
    }

    ucxx_trace_req("tag_recv_multi_header exiting, request: %p, tag: %lx, empty: %d",
                   bufferRequests.get(), tag, bufferRequests->bufferRequests.empty());
}

void tag_recv_multi_callback(std::shared_ptr<void> ucxxBufferRequests)
{
    auto bufferRequests = std::reinterpret_pointer_cast<UCXXBufferRequests>(ucxxBufferRequests);
    auto tag = bufferRequests->tag;

    ucxx_trace_req("tag_recv_multi_callback request: %p, tag: %lx",
                   bufferRequests.get(), tag);

    if (bufferRequests->bufferRequests.empty())
    {
        ucxx_trace_req("tag_recv_multi_callback first header, request: %p, tag: %lx",
                       bufferRequests.get(), tag);
        tag_recv_multi_header(ucxxBufferRequests);
    }
    else
    {
        const auto& request = bufferRequests->bufferRequests.back();
        auto header = Header(*bufferRequests->bufferRequests.back()->stringBuffer);

        ucxx_trace_req("tag_recv_multi_callback request: %p, tag: %lx, "
                       "num_requests: %lu, next: %d, request isCompleted: %d, "
                       "request status: %s",
                       bufferRequests.get(), tag, bufferRequests->bufferRequests.size(),
                       header.next, request->request->isCompleted(),
                       ucs_status_string(request->request->getStatus()));

        if (header.next)
            tag_recv_multi_header(ucxxBufferRequests);
        else
            tag_recv_multi_frames(ucxxBufferRequests);
    }
}

UCXXBufferRequestsPtr tag_recv_multi(
    std::shared_ptr<UCXXEndpoint> endpoint,
    ucp_tag_t tag)
{
    auto worker = UCXXEndpoint::getWorker(endpoint->getParent());

    auto bufferRequests = std::make_shared<UCXXBufferRequests>();
    bufferRequests->endpoint = endpoint;
    bufferRequests->tag = tag;

    ucxx_trace_req("tag_recv_multi request: %p, tag: %lx",
                   bufferRequests.get(), tag);

    tag_recv_multi_callback(bufferRequests);

    return bufferRequests;
}

// void tag_recv_multi_frames_callback(std::shared_ptr<void> data)
// {
//     std::cout << "tag_recv_multi_frames_callback: " << *std::static_pointer_cast<int>(data) << std::endl;
//     // free(data);
// }

// UCXXBufferRequestsPtr tag_recv_multi(
//     std::shared_ptr<UCXXEndpoint> endpoint,
//     ucp_tag_t tag)
// {
//     auto worker = UCXXEndpoint::getWorker(endpoint->getParent());
// 
//     auto bufferRequests = std::make_shared<UCXXBufferRequests>();
//     std::vector<Header> headers;
// 
//     do
//     {
//         auto bufferRequest = std::make_shared<UCXXBufferRequest>();
//         bufferRequest->stringBuffer = std::make_shared<std::string>(Header::dataSize(), 0);
//         // auto headerRequest = endpoint->tag_recv(headerSerialized.data(), headerSerialized.size(), tag);
//         auto cb_data = std::make_shared<int>(99);
//         bufferRequest->request = endpoint->tag_recv(bufferRequest->stringBuffer->data(),
//                                                     bufferRequest->stringBuffer->size(),
//                                                     //tag, (void*)tag_recv_multi_frames_callback, (void*)cb_data);
//                                                     tag, (void*)tag_recv_multi_frames_callback, std::static_pointer_cast<void>(cb_data));
//                                                     // tag, (void*)tag_recv_multi_frames_callback, std::dynamic_pointer_cast<void>(bufferRequests));
//                                                     // tag);
//         waitSingleRequest(worker, bufferRequest->request);
//         headers.push_back(Header(*bufferRequest->stringBuffer));
//         // headers.push_back(Header(headerSerialized));
//     } while (headers.back().next);
// 
//     for (auto& h : headers)
//     {
//         for (size_t i = 0; i < h.nframes; ++i)
//         {
//             auto bufferRequest = std::make_shared<UCXXBufferRequest>();
//             auto buf = allocateBuffer(h.isCUDA[i], h.size[i]);
//             bufferRequest->request = endpoint->tag_recv(buf->data(), buf->getSize(), tag);
//             bufferRequest->pyBuffer = std::move(buf);
//             bufferRequests->bufferRequests.push_back(bufferRequest);
//         }
//     }
// 
//     return bufferRequests;
// }

std::vector<std::unique_ptr<UCXXPyBuffer>> tag_recv_multi_b(
    std::shared_ptr<UCXXEndpoint> endpoint,
    ucp_tag_t tag)
{
    auto worker = UCXXEndpoint::getWorker(endpoint->getParent());

    auto requests = tag_recv_multi(endpoint, tag);

    std::vector<std::shared_ptr<UCXXRequest>> requestsOnly;
    std::vector<std::unique_ptr<UCXXPyBuffer>> recvBuffers;
    for (auto& br : requests->bufferRequests)
    {
        requestsOnly.push_back(br->request);
        recvBuffers.push_back(std::move(br->pyBuffer));
    }

    waitRequests(worker, requestsOnly);

    return recvBuffers;
}

UCXXBufferRequestsPtr tag_send_multi(
    std::shared_ptr<UCXXEndpoint> endpoint,
    std::vector<void*>& buffer,
    std::vector<size_t>& size,
    std::vector<int>& isCUDA,
    ucp_tag_t tag)
{
    auto bufferRequests = std::make_shared<UCXXBufferRequests>();

    size_t totalFrames = buffer.size();
    size_t totalHeaders = (totalFrames + HeaderFramesSize - 1) / HeaderFramesSize;

    for (size_t i = 0; i < totalHeaders; ++i)
    {
        bool hasNext = totalFrames > (i + 1) * HeaderFramesSize;
        size_t headerFrames = hasNext ? HeaderFramesSize : HeaderFramesSize - (HeaderFramesSize * (i + 1) - totalFrames);

        size_t idx = i * HeaderFramesSize;
        Header header(hasNext, headerFrames, (bool*)&isCUDA[idx], (size_t*)&size[idx]);
        auto serializedHeader = std::make_shared<std::string>(header.serialize());
        auto r = endpoint->tag_send(serializedHeader->data(), serializedHeader->size(), tag);

        auto bufferRequest = std::make_shared<UCXXBufferRequest>();
        bufferRequest->request = r;
        bufferRequest->stringBuffer = serializedHeader;
        bufferRequests->bufferRequests.push_back(bufferRequest);
    }

    for (size_t i = 0; i < buffer.size(); ++i)
    {
        auto r = endpoint->tag_send(buffer[i], size[i], tag);
        auto bufferRequest = std::make_shared<UCXXBufferRequest>();
        bufferRequest->request = r;
        bufferRequests->bufferRequests.push_back(bufferRequest);
    }

    bufferRequests->isFilled = true;
    ucxx_trace_req("tag_send_multi request: %p, tag: %lx, isFilled: %d",
                   bufferRequests.get(), tag, bufferRequests->isFilled);

    return bufferRequests;
}

void tag_send_multi_b(std::shared_ptr<UCXXEndpoint> endpoint,
                      std::vector<void*>& buffer,
                      std::vector<size_t>& size,
                      std::vector<int>& isCUDA,
                      ucp_tag_t tag)
{
    auto worker = UCXXEndpoint::getWorker(endpoint->getParent());

    auto requests = tag_send_multi(endpoint, buffer, size, isCUDA, tag);

    std::vector<std::shared_ptr<UCXXRequest>> requestsOnly;
    for (auto& br : requests->bufferRequests)
        requestsOnly.push_back(br->request);

    waitRequests(worker, requestsOnly);
}

}  // namespace ucxx
