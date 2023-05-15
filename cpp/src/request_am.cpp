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
#include <ucxx/internal/request_am.h>
#include <ucxx/request_am.h>

namespace ucxx {

std::shared_ptr<RequestAM> createRequestAMSend(
  std::shared_ptr<Endpoint> endpoint,
  void* buffer,
  size_t length,
  const bool enablePythonFuture                = false,
  RequestCallbackUserFunction callbackFunction = nullptr,
  RequestCallbackUserData callbackData         = nullptr)
{
  return std::shared_ptr<RequestAM>(
    new RequestAM(endpoint, buffer, length, enablePythonFuture, callbackFunction, callbackData));
}

std::shared_ptr<RequestAM> createRequestAMRecv(
  std::shared_ptr<Endpoint> endpoint,
  const bool enablePythonFuture                = false,
  RequestCallbackUserFunction callbackFunction = nullptr,
  RequestCallbackUserData callbackData         = nullptr)
{
  auto worker = endpoint->getWorker(endpoint->getParent());

  auto createRequest = [endpoint, enablePythonFuture, callbackFunction, callbackData]() {
    return std::shared_ptr<RequestAM>(
      new RequestAM(endpoint, enablePythonFuture, callbackFunction, callbackData));
  };
  return worker->getAmRecv(endpoint->getHandle(), createRequest);
}

RequestAM::RequestAM(std::shared_ptr<Endpoint> endpoint,
                     void* buffer,
                     size_t length,
                     const bool enablePythonFuture,
                     RequestCallbackUserFunction callbackFunction,
                     RequestCallbackUserData callbackData)
  : Request(endpoint,
            std::make_shared<DelayedSubmission>(true, buffer, length),
            std::string("amSend"),
            enablePythonFuture)
{
  _callback     = callbackFunction;
  _callbackData = callbackData;

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  _worker->registerDelayedSubmission(
    std::bind(std::mem_fn(&Request::populateDelayedSubmission), this));
}

RequestAM::RequestAM(std::shared_ptr<Component> endpointOrWorker,
                     const bool enablePythonFuture,
                     RequestCallbackUserFunction callbackFunction,
                     RequestCallbackUserData callbackData)
  : Request(endpointOrWorker, nullptr, std::string("amRecv"), enablePythonFuture)
{
  _callback     = callbackFunction;
  _callbackData = callbackData;
}

static void _amSendCallback(void* request, ucs_status_t status, void* user_data)
{
  Request* req = reinterpret_cast<Request*>(user_data);
  ucxx_trace_req_f(req->getOwnerString().c_str(), request, "amSend", "_amSendCallback");
  req->callback(request, status);
}

static void _recvCompletedCallback(void* request,
                                   ucs_status_t status,
                                   size_t length,
                                   void* user_data)
{
  internal::RecvAmMessage* recvAmMessage = static_cast<internal::RecvAmMessage*>(user_data);
  ucxx_trace_req_f(
    recvAmMessage->_request->getOwnerString().c_str(), request, "amRecv", "amRecvCallback");
  recvAmMessage->callback(request, status);
}

ucs_status_t RequestAM::recvCallback(void* arg,
                                     const void* header,
                                     size_t header_length,
                                     void* data,
                                     size_t length,
                                     const ucp_am_recv_param_t* param)
{
  internal::AmData* amData = reinterpret_cast<internal::AmData*>(arg);
  auto worker              = amData->_worker;
  auto& recvPool           = amData->_recvPool;
  auto& recvWait           = amData->_recvWait;

  if ((param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP) == 0)
    ucxx_error("UCP_AM_RECV_ATTR_FIELD_REPLY_EP not set");

  ucp_ep_h ep = param->reply_ep;
  recvPool.try_emplace(ep, std::queue<std::shared_ptr<RequestAM>>());

  bool is_rndv = param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV;

  std::shared_ptr<Buffer> buf{nullptr};
  auto allocatorType = *reinterpret_cast<const ucs_memory_type_t*>(header);

  std::shared_ptr<RequestAM> req{nullptr};

  {
    std::lock_guard<std::mutex> lock(amData->_mutex);

    if (recvWait.find(ep) != recvWait.end() && !recvWait[ep].empty()) {
      req = recvWait.at(ep).front();
      recvWait.at(ep).pop();
      ucxx_trace_req("amRecv recvWait: %p", req.get());
    } else {
      req = std::shared_ptr<RequestAM>(
        new RequestAM(worker, worker->isFutureEnabled(), nullptr, nullptr));
      recvPool.at(ep).push(req);
      ucxx_trace_req("amRecv recvPool: %p", req.get());
    }
  }

  if (is_rndv) {
    if (allocatorType == UCS_MEMORY_TYPE_HOST || allocatorType == UCS_MEMORY_TYPE_CUDA) {
      buf = amData->_allocators.at(allocatorType)(length);
    } else {
      ucxx_debug("Unsupported memory type %d", allocatorType);
      internal::RecvAmMessage recvAmMessage(amData, ep, req, nullptr);
      recvAmMessage.callback(nullptr, UCS_ERR_UNSUPPORTED);
      return UCS_ERR_UNSUPPORTED;
    }

    auto recvAmMessage = std::make_shared<internal::RecvAmMessage>(amData, ep, req, buf);

    ucp_request_param_t request_param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                                         UCP_OP_ATTR_FIELD_USER_DATA |
                                                         UCP_OP_ATTR_FLAG_NO_IMM_CMPL,
                                         .cb        = {.recv_am = _recvCompletedCallback},
                                         .user_data = recvAmMessage.get()};

    ucs_status_ptr_t status =
      ucp_am_recv_data_nbx(worker->getHandle(), data, buf->data(), length, &request_param);

    ucxx_debug("am rndv: ep %p len %lu", ep, length);

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
    ucxx_debug("am eager copying %lu bytes with ep %p", length, ep);

    buf = amData->_allocators.at(UCS_MEMORY_TYPE_HOST)(length);
    if (length > 0) memcpy(buf->data(), data, length);

    internal::RecvAmMessage recvAmMessage(amData, ep, req, buf);
    recvAmMessage.callback(nullptr, UCS_OK);
    return UCS_OK;
  }
}

std::shared_ptr<Buffer> RequestAM::getRecvBuffer() { return _buffer; }

void RequestAM::request()
{
  static const ucp_tag_t tagMask = -1;

  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_FLAGS |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .flags     = UCP_AM_SEND_FLAG_REPLY,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = this};

  // TODO: add CUDA support
  _sendHeader = UCS_MEMORY_TYPE_HOST;

  if (_delayedSubmission->_send) {
    param.cb.send = _amSendCallback;
    _request      = ucp_am_send_nbx(_endpoint->getHandle(),
                               0,
                               &_sendHeader,
                               sizeof(_sendHeader),
                               _delayedSubmission->_buffer,
                               _delayedSubmission->_length,
                               &param);
  } else {
    throw ucxx::UnsupportedError(
      "Receiving active messages must be handled by the worker's callback");
  }
}

void RequestAM::populateDelayedSubmission()
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
