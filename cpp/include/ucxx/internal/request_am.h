/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>

#include <ucp/api/ucp.h>

#include <ucxx/typedefs.h>

namespace ucxx {

class Buffer;
class InflightRequests;
class RequestAM;
class Request;
class Worker;

namespace internal {

class AmData;

class RecvAmMessage {
 public:
  internal::AmData* _amData{nullptr};
  ucp_ep_h _ep{nullptr};
  std::shared_ptr<RequestAM> _request{nullptr};
  std::shared_ptr<Buffer> _buffer{nullptr};

  RecvAmMessage(internal::AmData* amData,
                ucp_ep_h ep,
                std::shared_ptr<RequestAM> request,
                std::shared_ptr<Buffer> buffer);

  void setUcpRequest(void* request);

  void callback(void* request, ucs_status_t status);
};

typedef std::unordered_map<ucp_ep_h, std::queue<std::shared_ptr<RequestAM>>> AmPoolType;
typedef std::unordered_map<RequestAM*, std::shared_ptr<RecvAmMessage>> RecvAmMessageMapType;

class AmData {
 public:
  std::shared_ptr<Worker> _worker{nullptr};
  AmPoolType _recvPool{};
  AmPoolType _recvWait{};
  RecvAmMessageMapType _recvAmMessageMap{};
  std::mutex _mutex{};
  std::function<void(std::shared_ptr<Request>)> _registerInflightRequest{};
  std::unordered_map<ucs_memory_type_t, AmAllocatorType> _allocators{};
};

}  // namespace internal

}  // namespace ucxx
