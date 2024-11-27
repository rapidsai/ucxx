/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucs/memory/memory_type.h>
#include <ucxx/buffer.h>
#include <ucxx/delayed_submission.h>
#include <ucxx/internal/request_am.h>
#include <ucxx/request_am.h>
#include <ucxx/typedefs.h>

namespace ucxx {

AmReceiverCallbackInfo::AmReceiverCallbackInfo(const AmReceiverCallbackOwnerType owner,
                                               AmReceiverCallbackIdType id)
  : owner(owner), id(id)
{
}

typedef std::string AmHeaderSerialized;

struct AmHeader {
  ucs_memory_type_t memoryType;
  std::optional<AmReceiverCallbackInfo> receiverCallbackInfo;

  static AmHeader deserialize(const std::string_view serialized)
  {
    size_t offset{0};

    auto decode = [&offset, &serialized](void* data, size_t bytes) {
      memcpy(data, serialized.data() + offset, bytes);
      offset += bytes;
    };

    ucs_memory_type_t memoryType;
    decode(&memoryType, sizeof(memoryType));

    bool hasReceiverCallback{false};
    decode(&hasReceiverCallback, sizeof(hasReceiverCallback));

    if (hasReceiverCallback) {
      size_t ownerSize{0};
      decode(&ownerSize, sizeof(ownerSize));

      auto owner = AmReceiverCallbackOwnerType(ownerSize, 0);
      decode(owner.data(), ownerSize);

      AmReceiverCallbackIdType id{};
      decode(&id, sizeof(id));

      return AmHeader{.memoryType           = memoryType,
                      .receiverCallbackInfo = AmReceiverCallbackInfo(owner, id)};
    }

    return AmHeader{.memoryType = memoryType, .receiverCallbackInfo = std::nullopt};
  }

  const AmHeaderSerialized serialize() const
  {
    size_t offset{0};
    bool hasReceiverCallback{static_cast<bool>(receiverCallbackInfo)};
    const size_t ownerSize = (receiverCallbackInfo) ? receiverCallbackInfo->owner.size() : 0;
    const size_t amReceiverCallbackInfoSize =
      (receiverCallbackInfo) ? sizeof(ownerSize) + ownerSize + sizeof(receiverCallbackInfo->id) : 0;
    const size_t totalSize =
      sizeof(memoryType) + sizeof(hasReceiverCallback) + amReceiverCallbackInfoSize;
    std::string serialized(totalSize, 0);

    auto encode = [&offset, &serialized](void const* data, size_t bytes) {
      memcpy(serialized.data() + offset, data, bytes);
      offset += bytes;
    };

    encode(&memoryType, sizeof(memoryType));
    encode(&hasReceiverCallback, sizeof(hasReceiverCallback));
    if (hasReceiverCallback) {
      encode(&ownerSize, sizeof(ownerSize));
      encode(receiverCallbackInfo->owner.c_str(), ownerSize);
      encode(&receiverCallbackInfo->id, sizeof(receiverCallbackInfo->id));
    }

    return serialized;
  }
};

std::shared_ptr<RequestAm> createRequestAm(
  std::shared_ptr<Endpoint> endpoint,
  const std::variant<data::AmSend, data::AmReceive> requestData,
  const bool enablePythonFuture                = false,
  RequestCallbackUserFunction callbackFunction = nullptr,
  RequestCallbackUserData callbackData         = nullptr)
{
  std::shared_ptr<RequestAm> req = std::visit(
    data::dispatch{
      [endpoint, enablePythonFuture, callbackFunction, callbackData](data::AmSend amSend) {
        auto req = std::shared_ptr<RequestAm>(new RequestAm(
          endpoint, amSend, "amSend", enablePythonFuture, callbackFunction, callbackData));

        // A delayed notification request is not populated immediately, instead it is
        // delayed to allow the worker progress thread to set its status, and more
        // importantly the Python future later on, so that we don't need the GIL here.
        req->_worker->registerDelayedSubmission(
          req, std::bind(std::mem_fn(&Request::populateDelayedSubmission), req.get()));

        return req;
      },
      [endpoint, enablePythonFuture, callbackFunction, callbackData](data::AmReceive amReceive) {
        auto worker = endpoint->getWorker();

        auto createRequest = [endpoint,
                              amReceive,
                              enablePythonFuture,
                              callbackFunction,
                              callbackData]() {
          return std::shared_ptr<RequestAm>(new RequestAm(
            endpoint, amReceive, "amReceive", enablePythonFuture, callbackFunction, callbackData));
        };
        return worker->getAmRecv(endpoint->getHandle(), createRequest);
      },
    },
    requestData);

  return req;
}

RequestAm::RequestAm(std::shared_ptr<Component> endpointOrWorker,
                     const std::variant<data::AmSend, data::AmReceive> requestData,
                     const std::string operationName,
                     const bool enablePythonFuture,
                     RequestCallbackUserFunction callbackFunction,
                     RequestCallbackUserData callbackData)
  : Request(endpointOrWorker,
            data::getRequestData(requestData),
            operationName,
            enablePythonFuture,
            callbackFunction,
            callbackData)
{
  std::visit(data::dispatch{
               [this](data::AmSend amSend) {
                 if (_endpoint == nullptr)
                   throw ucxx::Error("An endpoint is required to send active messages");
               },
               [](data::AmReceive amReceive) {},
             },
             requestData);
}

void RequestAm::cancel()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  if (_status == UCS_INPROGRESS) {
    /**
     * This is needed to ensure AM requests are cancelable, since they do not
     * use the `_request`, thus `ucp_request_cancel()` cannot cancel them.
     */
    setStatus(UCS_ERR_CANCELED);
  } else {
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "already completed with status: %d (%s)",
                     _status,
                     ucs_status_string(_status));
  }
}

static void _amSendCallback(void* request, ucs_status_t status, void* user_data)
{
  Request* req = reinterpret_cast<Request*>(user_data);
  ucxx_trace_req_f(req->getOwnerString().c_str(), nullptr, request, "amSend", "_amSendCallback");
  req->callback(request, status);
}

static void _recvCompletedCallback(void* request,
                                   ucs_status_t status,
                                   size_t length,
                                   void* user_data)
{
  internal::RecvAmMessage* recvAmMessage = static_cast<internal::RecvAmMessage*>(user_data);
  ucxx_trace_req_f(recvAmMessage->_request->getOwnerString().c_str(),
                   nullptr,
                   request,
                   "amRecv",
                   "amRecvCallback");
  recvAmMessage->callback(request, status);
}

ucs_status_t RequestAm::recvCallback(void* arg,
                                     const void* header,
                                     size_t header_length,
                                     void* data,
                                     size_t length,
                                     const ucp_am_recv_param_t* param)
{
  internal::AmData* amData = static_cast<internal::AmData*>(arg);
  auto worker              = amData->_worker.lock();
  auto& ownerString        = amData->_ownerString;
  auto& recvPool           = amData->_recvPool;
  auto& recvWait           = amData->_recvWait;

  if ((param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP) == 0)
    ucxx_error("UCP_AM_RECV_ATTR_FIELD_REPLY_EP not set");

  ucp_ep_h ep = param->reply_ep;

  bool is_rndv = param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV;

  std::shared_ptr<Buffer> buf{nullptr};
  auto amHeader =
    AmHeader::deserialize(std::string_view(static_cast<const char*>(header), header_length));
  auto receiverCallback = [&amHeader, &amData]() {
    if (amHeader.receiverCallbackInfo) {
      try {
        return amData->_receiverCallbacks.at(amHeader.receiverCallbackInfo->owner)
          .at(amHeader.receiverCallbackInfo->id);
      } catch (std::out_of_range) {
        ucxx_error("No AM receiver callback registered for owner '%s' with id %lu",
                   std::string(amHeader.receiverCallbackInfo->owner).data(),
                   amHeader.receiverCallbackInfo->id);
      }
    }
    return AmReceiverCallbackType();
  }();

  std::shared_ptr<RequestAm> req{nullptr};

  {
    std::lock_guard<std::mutex> lock(amData->_mutex);

    auto reqs = recvWait.find(ep);
    if (amHeader.receiverCallbackInfo) {
      req = std::shared_ptr<RequestAm>(new RequestAm(
        worker, data::AmReceive(), "amReceive", worker->isFutureEnabled(), nullptr, nullptr));
      ucxx_trace_req_f(ownerString.c_str(), req.get(), nullptr, "amRecv", "receiverCallback");
    } else if (reqs != recvWait.end() && !reqs->second.empty()) {
      req = reqs->second.front();
      reqs->second.pop();
      ucxx_trace_req_f(ownerString.c_str(), req.get(), nullptr, "amRecv", "recvWait");
    } else {
      req             = std::shared_ptr<RequestAm>(new RequestAm(
        worker, data::AmReceive(), "amReceive", worker->isFutureEnabled(), nullptr, nullptr));
      auto [queue, _] = recvPool.try_emplace(ep, std::queue<std::shared_ptr<RequestAm>>());
      queue->second.push(req);
      ucxx_trace_req_f(ownerString.c_str(), req.get(), nullptr, "amRecv", "recvPool");
    }
  }

  if (is_rndv) {
    if (amData->_allocators.find(amHeader.memoryType) == amData->_allocators.end()) {
      // TODO: Is a hard failure better?
      // ucxx_debug("Unsupported memory type %d", amHeader.memoryType);
      // internal::RecvAmMessage recvAmMessage(amData, ep, req, nullptr);
      // recvAmMessage.callback(nullptr, UCS_ERR_UNSUPPORTED);
      // return UCS_ERR_UNSUPPORTED;

      ucxx_trace_req("No allocator registered for memory type %u, falling back to host memory.",
                     amHeader.memoryType);
      amHeader.memoryType = UCS_MEMORY_TYPE_HOST;
    }

    try {
      buf = amData->_allocators.at(amHeader.memoryType)(length);
    } catch (const std::exception& e) {
      ucxx_debug("Exception calling allocator: %s", e.what());
    }

    auto recvAmMessage =
      std::make_shared<internal::RecvAmMessage>(amData, ep, req, buf, receiverCallback);

    ucp_request_param_t requestParam = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                                        UCP_OP_ATTR_FIELD_USER_DATA |
                                                        UCP_OP_ATTR_FLAG_NO_IMM_CMPL,
                                        .cb        = {.recv_am = _recvCompletedCallback},
                                        .user_data = recvAmMessage.get()};

    if (buf == nullptr) {
      ucxx_debug("Failed to allocate %lu bytes of memory", length);
      recvAmMessage->_request->setStatus(UCS_ERR_NO_MEMORY);
      return UCS_ERR_NO_MEMORY;
    }

    ucs_status_ptr_t status =
      ucp_am_recv_data_nbx(worker->getHandle(), data, buf->data(), length, &requestParam);

    if (req->_enablePythonFuture)
      ucxx_trace_req_f(ownerString.c_str(),
                       req.get(),
                       status,
                       "amRecv rndv",
                       "recvCallback, ep: %p, buffer: %p, size: %lu, future: %p, future handle: %p",
                       ep,
                       buf->data(),
                       length,
                       req->_future.get(),
                       req->_future->getHandle());
    else
      ucxx_trace_req_f(ownerString.c_str(),
                       req.get(),
                       status,
                       "amRecv rndv",
                       "recvCallback, ep: %p, buffer: %p, size: %lu",
                       ep,
                       buf->data(),
                       length);

    if (req->isCompleted()) {
      // The request completed/errored immediately
      ucs_status_t s = UCS_PTR_STATUS(status);
      recvAmMessage->callback(nullptr, s);

      return s;
    } else {
      // The request will be handled by the callback
      recvAmMessage->setUcpRequest(status);
      amData->_registerInflightRequest(req);

      {
        std::lock_guard<std::mutex> lock(amData->_mutex);
        amData->_recvAmMessageMap.emplace(req.get(), recvAmMessage);
      }

      return UCS_INPROGRESS;
    }
  } else {
    buf = amData->_allocators.at(UCS_MEMORY_TYPE_HOST)(length);

    internal::RecvAmMessage recvAmMessage(amData, ep, req, buf, receiverCallback);
    if (buf == nullptr) {
      ucxx_debug("Failed to allocate %lu bytes of memory", length);
      recvAmMessage._request->setStatus(UCS_ERR_NO_MEMORY);
      return UCS_ERR_NO_MEMORY;
    }

    if (length > 0) memcpy(buf->data(), data, length);

    if (req->_enablePythonFuture)
      ucxx_trace_req_f(ownerString.c_str(),
                       req.get(),
                       nullptr,
                       "amRecv eager",
                       "recvCallback, ep: %p, buffer: %p, size: %lu, future: %p, future handle: %p",
                       ep,
                       buf->data(),
                       length,
                       req->_future.get(),
                       req->_future->getHandle());
    else
      ucxx_trace_req_f(ownerString.c_str(),
                       req.get(),
                       nullptr,
                       "amRecv eager",
                       "recvCallback, ep: %p, buffer: %p, size: %lu",
                       ep,
                       buf->data(),
                       length);

    recvAmMessage.callback(nullptr, UCS_OK);
    return UCS_OK;
  }
}

std::shared_ptr<Buffer> RequestAm::getRecvBuffer()
{
  return std::visit(
    data::dispatch{
      [](data::AmReceive amReceive) { return amReceive._buffer; },
      [](auto) -> std::shared_ptr<Buffer> { throw std::runtime_error("Unreachable"); },
    },
    _requestData);
}

void RequestAm::request()
{
  std::visit(
    data::dispatch{
      [this](data::AmSend amSend) {
        ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                                     UCP_OP_ATTR_FIELD_FLAGS |
                                                     UCP_OP_ATTR_FIELD_USER_DATA,
                                     .flags = UCP_AM_SEND_FLAG_REPLY | UCP_AM_SEND_FLAG_COPY_HEADER,
                                     .datatype  = ucp_dt_make_contig(1),
                                     .user_data = this};

        param.cb.send         = _amSendCallback;
        AmHeader header       = {.memoryType           = amSend._memoryType,
                                 .receiverCallbackInfo = amSend._receiverCallbackInfo};
        auto headerSerialized = header.serialize();
        void* request         = ucp_am_send_nbx(_endpoint->getHandle(),
                                        0,
                                        headerSerialized.data(),
                                        headerSerialized.size(),
                                        amSend._buffer,
                                        amSend._length,
                                        &param);

        std::lock_guard<std::recursive_mutex> lock(_mutex);
        _request = request;
      },
      [](auto) { throw ucxx::UnsupportedError("Only send active messages can call request()"); },
    },
    _requestData);
}

void RequestAm::populateDelayedSubmission()
{
  bool terminate =
    std::visit(data::dispatch{
                 [this](data::AmSend amSend) {
                   if (_endpoint->getHandle() == nullptr) {
                     ucxx_warn("Endpoint was closed before message could be sent");
                     Request::callback(this, UCS_ERR_CANCELED);
                     return true;
                   }
                   return false;
                 },
                 [](auto) -> decltype(terminate) { throw std::runtime_error("Unreachable"); },
               },
               _requestData);
  if (terminate) return;

  request();

  auto log = [this](const void* buffer, const size_t length, const ucs_memory_type_t memoryType) {
    if (_enablePythonFuture)
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "populateDelayedSubmission, buffer %p, size %lu, memoryType: %u, future %p, "
                       "future handle %p, ",
                       buffer,
                       length,
                       memoryType,
                       _future.get(),
                       _future->getHandle());
    else
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "populateDelayedSubmission, buffer %p, size %lu, memoryType: %u",
                       buffer,
                       length,
                       memoryType);
  };

  std::visit(data::dispatch{
               [this, &log](data::AmSend amSend) {
                 log(amSend._buffer, amSend._length, amSend._memoryType);
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _requestData);

  process();
}

}  // namespace ucxx
