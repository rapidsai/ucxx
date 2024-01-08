/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/buffer.h>
#include <ucxx/delayed_submission.h>
#include <ucxx/request_mem.h>

namespace ucxx {

std::shared_ptr<RequestMem> createRequestMem(std::shared_ptr<Endpoint> endpoint,
                                             bool send,
                                             void* buffer,
                                             size_t length,
                                             uint64_t remote_addr,
                                             ucp_rkey_h rkey,
                                             const bool enablePythonFuture                = false,
                                             RequestCallbackUserFunction callbackFunction = nullptr,
                                             RequestCallbackUserData callbackData         = nullptr)
{
  auto req = std::shared_ptr<RequestMem>(new RequestMem(endpoint,
                                                        send,
                                                        buffer,
                                                        length,
                                                        remote_addr,
                                                        rkey,
                                                        enablePythonFuture,
                                                        callbackFunction,
                                                        callbackData));

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  req->_worker->registerDelayedSubmission(
    req, std::bind(std::mem_fn(&Request::populateDelayedSubmission), req.get()));

  return req;
}

RequestMem::RequestMem(std::shared_ptr<Endpoint> endpoint,
                       bool send,
                       void* buffer,
                       size_t length,
                       uint64_t remote_addr,
                       ucp_rkey_h rkey,
                       const bool enablePythonFuture,
                       RequestCallbackUserFunction callbackFunction,
                       RequestCallbackUserData callbackData)
  : Request(endpoint,
            std::make_shared<DelayedSubmission>(send, buffer, length, 0),
            std::string(send ? "memSend" : "memRecv"),
            enablePythonFuture),
    _remote_addr(remote_addr),
    _rkey(rkey)
{
  if (_endpoint == nullptr)
    throw ucxx::Error("An endpoint is required to perform remote memory put/get messages");

  _callback     = callbackFunction;
  _callbackData = callbackData;
}

// static void _memSendCallback(void* request, ucs_status_t status, void* user_data)
// {
//   Request* req = reinterpret_cast<Request*>(user_data);
//   ucxx_trace_req_f(req->getOwnerString().c_str(), request, "memSend", "_memSendCallback");
//   req->callback(request, status);
// }

// static void _recvCompletedCallback(void* request,
//                                    ucs_status_t status,
//                                    size_t length,
//                                    void* user_data)
// {
//   internal::RecvMemMessage* recvMemMessage = static_cast<internal::RecvMemMessage*>(user_data);
//   ucxx_trace_req_f(
//     recvMemMessage->_request->getOwnerString().c_str(), request, "memRecv", "memRecvCallback");
//   recvMemMessage->callback(request, status);
// }

// ucs_status_t RequestMem::recvCallback(void* arg,
//                                       const void* header,
//                                       size_t header_length,
//                                       void* data,
//                                       size_t length,
//                                       const ucp_am_recv_param_t* param)
// {
//   internal::AmData* memData = static_cast<internal::MemData*>(arg);
//   auto worker               = memData->_worker.lock();
//   auto& ownerString         = memData->_ownerString;
//   auto& recvPool            = memData->_recvPool;
//   auto& recvWait            = memData->_recvWait;

//   if ((param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP) == 0)
//     ucxx_error("UCP_AM_RECV_ATTR_FIELD_REPLY_EP not set");

//   ucp_ep_h ep = param->reply_ep;

//   bool is_rndv = param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV;

//   std::shared_ptr<Buffer> buf{nullptr};
//   auto allocatorType = *static_cast<const ucs_memory_type_t*>(header);

//   std::shared_ptr<RequestMem> req{nullptr};

//   {
//     std::lock_guard<std::mutex> lock(memData->_mutex);

//     auto reqs = recvWait.find(ep);
//     if (reqs != recvWait.end() && !reqs->second.empty()) {
//       req = reqs->second.front();
//       reqs->second.pop();
//       ucxx_trace_req("memRecv recvWait: %p", req.get());
//     } else {
//       req = std::shared_ptr<RequestMem>(
//         new RequestMem(worker, worker->isFutureEnabled(), nullptr, nullptr));
//       auto [queue, _] = recvPool.try_emplace(ep, std::queue<std::shared_ptr<RequestMem>>());
//       queue->second.push(req);
//       ucxx_trace_req("memRecv recvPool: %p", req.get());
//     }
//   }

//   if (is_rndv) {
//     if (memData->_allocators.find(allocatorType) == memData->_allocators.end()) {
//       // TODO: Is a hard failure better?
//       // ucxx_debug("Unsupported memory type %d", allocatorType);
//       // internal::RecvMemMessage recvMemMessage(memData, ep, req, nullptr);
//       // recvMemMessage.callback(nullptr, UCS_ERR_UNSUPPORTED);
//       // return UCS_ERR_UNSUPPORTED;

//       ucxx_trace_req("No allocator registered for memory type %d, falling back to host memory.",
//                      allocatorType);
//       allocatorType = UCS_MEMORY_TYPE_HOST;
//     }

//     std::shared_ptr<Buffer> buf = memData->_allocators.at(allocatorType)(length);

//     auto recvMemMessage = std::make_shared<internal::RecvMemMessage>(memData, ep, req, buf);

//     ucp_request_param_t request_param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
//                                                          UCP_OP_ATTR_FIELD_USER_DATA |
//                                                          UCP_OP_ATTR_FLAG_NO_IMM_CMPL,
//                                          .cb        = {.recv_am = _recvCompletedCallback},
//                                          .user_data = recvMemMessage.get()};

//     ucs_status_ptr_t status =
//       ucp_am_recv_data_nbx(worker->getHandle(), data, buf->data(), length, &request_param);

//     if (req->_enablePythonFuture)
//       ucxx_trace_req_f(ownerString.c_str(),
//                        status,
//                        "memRecv rndv",
//                        "ep %p, buffer %p, size %lu, future %p, future handle %p, recvCallback",
//                        ep,
//                        buf->data(),
//                        length,
//                        req->_future.get(),
//                        req->_future->getHandle());
//     else
//       ucxx_trace_req_f(ownerString.c_str(),
//                        status,
//                        "memRecv rndv",
//                        "ep %p, buffer %p, size %lu, recvCallback",
//                        ep,
//                        buf->data(),
//                        length);

//     if (req->isCompleted()) {
//       // The request completed/errored immediately
//       ucs_status_t s = UCS_PTR_STATUS(status);
//       recvMemMessage->callback(nullptr, s);

//       return s;
//     } else {
//       // The request will be handled by the callback
//       recvMemMessage->setUcpRequest(status);
//       memData->_registerInflightRequest(req);

//       {
//         std::lock_guard<std::mutex> lock(memData->_mutex);
//         memData->_recvMemMessageMap.emplace(req.get(), recvMemMessage);
//       }

//       return UCS_INPROGRESS;
//     }
//   } else {
//     std::shared_ptr<Buffer> buf = memData->_allocators.at(UCS_MEMORY_TYPE_HOST)(length);
//     if (length > 0) memcpy(buf->data(), data, length);

//     if (req->_enablePythonFuture)
//       ucxx_trace_req_f(ownerString.c_str(),
//                        nullptr,
//                        "memRecv eager",
//                        "ep: %p, buffer %p, size %lu, future %p, future handle %p, recvCallback",
//                        ep,
//                        buf->data(),
//                        length,
//                        req->_future.get(),
//                        req->_future->getHandle());
//     else
//       ucxx_trace_req_f(ownerString.c_str(),
//                        nullptr,
//                        "memRecv eager",
//                        "ep: %p, buffer %p, size %lu, recvCallback",
//                        ep,
//                        buf->data(),
//                        length);

//     internal::RecvMemMessage recvMemMessage(memData, ep, req, buf);
//     recvMemMessage.callback(nullptr, UCS_OK);
//     return UCS_OK;
//   }
// }

void RequestMem::memPutCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), request, "memSend", "memPutCallback");
  return req->callback(request, status);
}

void RequestMem::memGetCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), request, "memRecv", "memGetCallback");
  return req->callback(request, status);
}

void RequestMem::request()
{
  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_FLAGS |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .flags     = UCP_AM_SEND_FLAG_REPLY,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = this};

  void* request = nullptr;

  if (_delayedSubmission->_send) {
    param.cb.send = memPutCallback;
    request       = ucp_put_nbx(_endpoint->getHandle(),
                          _delayedSubmission->_buffer,
                          _delayedSubmission->_length,
                          _remote_addr,
                          _rkey,
                          &param);

    std::lock_guard<std::recursive_mutex> lock(_mutex);
    _request = request;
  } else {
    param.cb.send = memGetCallback;
    request       = ucp_get_nbx(_endpoint->getHandle(),
                          _delayedSubmission->_buffer,
                          _delayedSubmission->_length,
                          _remote_addr,
                          _rkey,
                          &param);
  }

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  _request = request;
}

void RequestMem::populateDelayedSubmission()
{
  request();

  if (_enablePythonFuture)
    ucxx_trace_req_f(_ownerString.c_str(),
                     _request,
                     _operationName.c_str(),
                     "buffer %p, size %lu, future %p, future handle %p, populateDelayedSubmission",
                     _delayedSubmission->_buffer,
                     _delayedSubmission->_length,
                     _future.get(),
                     _future->getHandle());
  else
    ucxx_trace_req_f(_ownerString.c_str(),
                     _request,
                     _operationName.c_str(),
                     "buffer %p, size %lu, populateDelayedSubmission",
                     _delayedSubmission->_buffer,
                     _delayedSubmission->_length);

  process();
}

}  // namespace ucxx
