/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request_data.h>
#include <ucxx/request_tag.h>

namespace ucxx {

std::shared_ptr<RequestTag> createRequestTag(
  std::shared_ptr<Component> endpointOrWorker,
  const std::variant<data::TagSend, data::TagReceive> requestData,
  const bool enablePythonFuture                = false,
  RequestCallbackUserFunction callbackFunction = nullptr,
  RequestCallbackUserData callbackData         = nullptr)
{
  std::shared_ptr<RequestTag> req =
    std::visit(data::dispatch{
                 [&endpointOrWorker, &enablePythonFuture, &callbackFunction, &callbackData](
                   data::TagSend tagSend) {
                   return std::shared_ptr<RequestTag>(new RequestTag(endpointOrWorker,
                                                                     tagSend,
                                                                     "tagSend",
                                                                     enablePythonFuture,
                                                                     callbackFunction,
                                                                     callbackData));
                 },
                 [&endpointOrWorker, &enablePythonFuture, &callbackFunction, &callbackData](
                   data::TagReceive tagReceive) {
                   return std::shared_ptr<RequestTag>(new RequestTag(endpointOrWorker,
                                                                     tagReceive,
                                                                     "tagRecv",
                                                                     enablePythonFuture,
                                                                     callbackFunction,
                                                                     callbackData));
                 },
               },
               requestData);

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  req->_worker->registerDelayedSubmission(
    req, std::bind(std::mem_fn(&Request::populateDelayedSubmission), req.get()));

  return req;
}

RequestTag::RequestTag(std::shared_ptr<Component> endpointOrWorker,
                       const std::variant<data::TagSend, data::TagReceive> requestData,
                       const std::string operationName,
                       const bool enablePythonFuture,
                       RequestCallbackUserFunction callbackFunction,
                       RequestCallbackUserData callbackData)
  : Request(endpointOrWorker, data::getRequestData(requestData), operationName, enablePythonFuture)
{
  std::visit(data::dispatch{
               [this](data::TagSend tagSend) {
                 if (_endpoint == nullptr)
                   throw ucxx::Error("An endpoint is required to send tag messages");
               },
               [](data::TagReceive tagReceive) {},
             },
             requestData);

  _callback     = callbackFunction;
  _callbackData = callbackData;
}

void RequestTag::callback(void* request, ucs_status_t status, const ucp_tag_recv_info_t* info)
{
  // TODO: Decide on behavior. See https://github.com/rapidsai/ucxx/issues/104 .
  // if (status != UCS_ERR_CANCELED && info->length != _length) {
  //   status          = UCS_ERR_MESSAGE_TRUNCATED;
  //   const char* fmt = "length mismatch: %llu (got) != %llu (expected)";
  //   size_t len      = std::snprintf(nullptr, 0, fmt, info->length, _length);
  //   _status_msg     = std::string(len + 1, '\0');  // +1 for null terminator
  //   std::snprintf(_status_msg.data(), _status_msg.size(), fmt, info->length, _length);
  // }

  Request::callback(request, status);
}

void RequestTag::tagSendCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), request, "tagSend", "tagSendCallback");
  return req->callback(request, status);
}

void RequestTag::tagRecvCallback(void* request,
                                 ucs_status_t status,
                                 const ucp_tag_recv_info_t* info,
                                 void* arg)
{
  RequestTag* req = reinterpret_cast<RequestTag*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), request, "tagRecv", "tagRecvCallback");
  return req->callback(request, status, info);
}

void RequestTag::request()
{
  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_DATATYPE |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = this};
  void* request             = nullptr;

  std::visit(data::dispatch{
               [this, &request, &param](data::TagSend tagSend) {
                 param.cb.send = tagSendCallback;
                 request       = ucp_tag_send_nbx(
                   _endpoint->getHandle(), tagSend._buffer, tagSend._length, tagSend._tag, &param);
               },
               [this, &request, &param](data::TagReceive tagReceive) {
                 param.cb.recv = tagRecvCallback;
                 request       = ucp_tag_recv_nbx(_worker->getHandle(),
                                            tagReceive._buffer,
                                            tagReceive._length,
                                            tagReceive._tag,
                                            tagReceive._tagMask,
                                            &param);
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _requestData);

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  _request = request;
}

static void logPopulateDelayedSubmission() {}

void RequestTag::populateDelayedSubmission()
{
  bool terminate =
    std::visit(data::dispatch{
                 [this](data::TagSend tagSend) {
                   if (_endpoint->getHandle() == nullptr) {
                     ucxx_warn("Endpoint was closed before message could be sent");
                     Request::callback(this, UCS_ERR_CANCELED);
                     return true;
                   }
                   return false;
                 },
                 [this](data::TagReceive tagReceive) {
                   if (_worker->getHandle() == nullptr) {
                     ucxx_warn("Worker was closed before message could be received");
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

  auto log = [this](const void* buffer, const size_t length, const Tag tag, const TagMask tagMask) {
    if (_enablePythonFuture)
      ucxx_trace_req_f(
        _ownerString.c_str(),
        _request,
        _operationName.c_str(),
        "buffer: %p, size: %lu, tag 0x%lx, tagMask: 0x%lx, future %p, future handle %p, "
        "populateDelayedSubmission",
        buffer,
        length,
        tag,
        tagMask,
        _future.get(),
        _future->getHandle());
    else
      ucxx_trace_req_f(
        _ownerString.c_str(),
        _request,
        _operationName.c_str(),
        "buffer: %p, size: %lu, tag 0x%lx, tagMask: 0x%lx, populateDelayedSubmission",
        buffer,
        length,
        tag,
        tagMask);
  };

  std::visit(data::dispatch{
               [this, &log](data::TagSend tagSend) {
                 log(tagSend._buffer, tagSend._length, tagSend._tag, TagMaskFull);
               },
               [this, &log](data::TagReceive tagReceive) {
                 log(tagReceive._buffer, tagReceive._length, tagReceive._tag, tagReceive._tagMask);
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _requestData);

  process();
}

}  // namespace ucxx
