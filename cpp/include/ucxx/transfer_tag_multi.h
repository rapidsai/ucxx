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

std::vector<std::unique_ptr<UCXXPyBuffer>> tag_recv_multi_b(
    std::shared_ptr<UCXXEndpoint> endpoint,
    ucp_tag_t tag)
{
    auto worker = UCXXEndpoint::getWorker(endpoint->getParent());

    std::vector<std::shared_ptr<UCXXRequest>> requests;
    Header header;
    std::vector<Header> headers;
    std::vector<std::unique_ptr<UCXXPyBuffer>> recvBuffers;

    do
    {
        std::string headerSerialized(Header::dataSize(), 0);
        auto headerRequest = endpoint->tag_recv(headerSerialized.data(), headerSerialized.size(), tag);
        waitSingleRequest(worker, headerRequest);
        headers.push_back(Header(headerSerialized));
    } while (headers.back().next);

    for (auto& h : headers)
    {
        for (size_t i = 0; i < h.nframes; ++i)
        {
            auto buf = allocateBuffer(h.isCUDA[i], h.size[i]);
            requests.push_back(endpoint->tag_recv(buf->data(), buf->getSize(), tag));
            recvBuffers.push_back(std::move(buf));
        }
    }

    waitRequests(worker, requests);

    return recvBuffers;
}


struct UCXXBufferRequest
{
    std::shared_ptr<UCXXRequest> request{nullptr};
    std::shared_ptr<std::string> stringBuffer{nullptr};
};

std::vector<UCXXBufferRequest> tag_send_multi(
    std::shared_ptr<UCXXEndpoint> endpoint,
    std::vector<void*>& buffer,
    std::vector<size_t>& size,
    std::vector<int>& isCUDA,
    ucp_tag_t tag)
{
    std::vector<UCXXBufferRequest> requests;

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

        UCXXBufferRequest request = {
            .request = r,
            .stringBuffer = serializedHeader
        };
        requests.push_back(request);
    }

    for (size_t i = 0; i < buffer.size(); ++i)
    {
        auto r = endpoint->tag_send(buffer[i], size[i], tag);
        UCXXBufferRequest request = {
            .request = r,
        };
        requests.push_back(request);
    }

    return requests;
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
    for (auto& r: requests)
        requestsOnly.push_back(r.request);

    waitRequests(worker, requestsOnly);
}

}  // namespace ucxx
