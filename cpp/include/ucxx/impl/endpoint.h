/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint.h>
#include <ucxx/exception.h>
#include <ucxx/listener.h>
#include <ucxx/request_stream.h>
#include <ucxx/request_tag.h>
#include <ucxx/sockaddr_utils.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils.h>
#include <ucxx/worker.h>

namespace ucxx {

void EpParamsDeleter::operator()(ucp_ep_params_t* ptr)
{
  if (ptr != nullptr && ptr->field_mask & UCP_EP_PARAM_FIELD_SOCK_ADDR)
    sockaddr_utils_free(&ptr->sockaddr);
}

Endpoint::Endpoint(std::shared_ptr<Component> worker_or_listener,
                   std::unique_ptr<ucp_ep_params_t, EpParamsDeleter> params,
                   bool endpoint_error_handling)
  : _endpoint_error_handling{endpoint_error_handling}
{
  auto worker = Endpoint::getWorker(worker_or_listener);

  if (worker == nullptr || worker->get_handle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  setParent(worker_or_listener);

  _callbackData = std::make_unique<error_callback_data_t>((error_callback_data_t){
    .status = UCS_OK, .inflightRequests = _inflightRequests, .worker = worker});

  params->err_mode =
    (endpoint_error_handling ? UCP_ERR_HANDLING_MODE_PEER : UCP_ERR_HANDLING_MODE_NONE);
  params->err_handler.cb  = Endpoint::errorCallback;
  params->err_handler.arg = _callbackData.get();

  assert_ucs_status(ucp_ep_create(worker->get_handle(), params.get(), &_handle));
}

Endpoint::~Endpoint()
{
  if (_handle == nullptr) return;

  // Close the endpoint
  unsigned close_mode = UCP_EP_CLOSE_MODE_FORCE;
  if (_endpoint_error_handling and _callbackData->status != UCS_OK) {
    // We force close endpoint if endpoint error handling is enabled and
    // the endpoint status is not UCS_OK
    close_mode = UCP_EP_CLOSE_MODE_FORCE;
  }
  ucs_status_ptr_t status = ucp_ep_close_nb(_handle, close_mode);
  if (UCS_PTR_IS_PTR(status)) {
    auto worker = Endpoint::getWorker(_parent);
    while (ucp_request_check_status(status) == UCS_INPROGRESS)
      worker->progress();
    ucp_request_free(status);
  } else if (UCS_PTR_STATUS(status) != UCS_OK) {
    ucxx_error("Error while closing endpoint: %s", ucs_status_string(UCS_PTR_STATUS(status)));
  }
}

ucp_ep_h Endpoint::getHandle() { return _handle; }

bool Endpoint::isAlive() const
{
  if (!_endpoint_error_handling) return true;

  return _callbackData->status == UCS_OK;
}

void Endpoint::raiseOnError()
{
  ucs_status_t status = _callbackData->status;

  if (status == UCS_OK || !_endpoint_error_handling) return;

  std::string statusString{ucs_status_string(status)};
  std::stringstream errorMsgStream;
  errorMsgStream << "Endpoint " << std::hex << _handle << " error: " << statusString;

  if (status == UCS_ERR_CONNECTION_RESET)
    throw ConnectionResetError(errorMsgStream.str());
  else
    throw Error(errorMsgStream.str());
}

void Endpoint::setCloseCallback(std::function<void(void*)> closeCallback, void* closeCallbackArg)
{
  _callbackData->closeCallback    = closeCallback;
  _callbackData->closeCallbackArg = closeCallbackArg;
}

void Endpoint::registerInflightRequest(std::shared_ptr<Request> request)
{
  std::weak_ptr<Request> weak_req = request;
  _inflightRequests->insert({request.get(), weak_req});
}

void Endpoint::removeInflightRequest(Request* request)
{
  auto search = _inflightRequests->find(request);
  if (search != _inflightRequests->end()) _inflightRequests->erase(search);
}

std::shared_ptr<Request> Endpoint::stream_send(void* buffer, size_t length)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  auto request  = createRequestStream(endpoint, true, buffer, length);
  registerInflightRequest(request);
  return request;
}

std::shared_ptr<Request> Endpoint::stream_recv(void* buffer, size_t length)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  auto request  = createRequestStream(endpoint, false, buffer, length);
  registerInflightRequest(request);
  return request;
}

std::shared_ptr<Request> Endpoint::tag_send(
  void* buffer,
  size_t length,
  ucp_tag_t tag,
  const bool enablePythonFuture,
  std::function<void(std::shared_ptr<void>)> callbackFunction,
  std::shared_ptr<void> callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  auto request  = createRequestTag(
    endpoint, true, buffer, length, tag, enablePythonFuture, callbackFunction, callbackData);
  registerInflightRequest(request);
  return request;
}

std::shared_ptr<Request> Endpoint::tag_recv(
  void* buffer,
  size_t length,
  ucp_tag_t tag,
  const bool enablePythonFuture,
  std::function<void(std::shared_ptr<void>)> callbackFunction,
  std::shared_ptr<void> callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  auto request  = createRequestTag(
    endpoint, false, buffer, length, tag, enablePythonFuture, callbackFunction, callbackData);
  registerInflightRequest(request);
  return request;
}

std::shared_ptr<Worker> Endpoint::getWorker(std::shared_ptr<Component> workerOrListener)
{
  auto worker = std::dynamic_pointer_cast<Worker>(workerOrListener);
  if (worker == nullptr) {
    auto listener = std::dynamic_pointer_cast<Listener>(workerOrListener);
    worker        = std::dynamic_pointer_cast<Worker>(listener->getParent());
  }
  return worker;
}

void Endpoint::errorCallback(void* arg, ucp_ep_h ep, ucs_status_t status)
{
  error_callback_data_t* data = (error_callback_data_t*)arg;
  data->status            = status;
  data->worker->scheduleRequestCancel(data->inflightRequests);
  if (data->closeCallback) {
    data->closeCallback(data->closeCallbackArg);
    data->closeCallback    = nullptr;
    data->closeCallbackArg = nullptr;
  }

  // Connection reset and timeout often represent just a normal remote
  // endpoint disconnect, log only in debug mode.
  if (status == UCS_ERR_CONNECTION_RESET || status == UCS_ERR_ENDPOINT_TIMEOUT)
    ucxx_debug("Error callback for endpoint %p called with status %d: %s",
               ep,
               status,
               ucs_status_string(status));
  else
    ucxx_error("Error callback for endpoint %p called with status %d: %s",
               ep,
               status,
               ucs_status_string(status));
}

}  // namespace ucxx
