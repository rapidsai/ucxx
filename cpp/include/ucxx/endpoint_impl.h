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

UCXXEndpoint::UCXXEndpoint(std::shared_ptr<UCXXComponent> worker_or_listener,
                           std::unique_ptr<ucp_ep_params_t, EpParamsDeleter> params,
                           bool endpoint_error_handling)
  : _endpoint_error_handling{endpoint_error_handling}
{
  auto worker = UCXXEndpoint::getWorker(worker_or_listener);

  if (worker == nullptr || worker->get_handle() == nullptr)
    throw ucxx::UCXXError("Worker not initialized");

  setParent(worker_or_listener);

  _callbackData = std::make_unique<error_callback_data_t>((error_callback_data_t){
    .status = UCS_OK, .inflightRequests = _inflightRequests, .worker = worker});

  params->err_mode =
    (endpoint_error_handling ? UCP_ERR_HANDLING_MODE_PEER : UCP_ERR_HANDLING_MODE_NONE);
  params->err_handler.cb  = UCXXEndpoint::errorCallback;
  params->err_handler.arg = _callbackData.get();

  assert_ucs_status(ucp_ep_create(worker->get_handle(), params.get(), &_handle));
}

UCXXEndpoint::~UCXXEndpoint()
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
    auto worker = UCXXEndpoint::getWorker(_parent);
    while (ucp_request_check_status(status) == UCS_INPROGRESS)
      worker->progress();
    ucp_request_free(status);
  } else if (UCS_PTR_STATUS(status) != UCS_OK) {
    ucxx_error("Error while closing endpoint: %s", ucs_status_string(UCS_PTR_STATUS(status)));
  }
}

ucp_ep_h UCXXEndpoint::getHandle() { return _handle; }

bool UCXXEndpoint::isAlive() const
{
  if (!_endpoint_error_handling) return true;

  return _callbackData->status == UCS_OK;
}

void UCXXEndpoint::raiseOnError()
{
  ucs_status_t status = _callbackData->status;

  if (status == UCS_OK || !_endpoint_error_handling) return;

  std::string statusString{ucs_status_string(status)};
  std::stringstream errorMsgStream;
  errorMsgStream << "Endpoint " << std::hex << _handle << " error: " << statusString;

  if (status == UCS_ERR_CONNECTION_RESET)
    throw UCXXConnectionResetError(errorMsgStream.str());
  else
    throw UCXXError(errorMsgStream.str());
}

void UCXXEndpoint::setCloseCallback(std::function<void(void*)> closeCallback,
                                    void* closeCallbackArg)
{
  _callbackData->closeCallback    = closeCallback;
  _callbackData->closeCallbackArg = closeCallbackArg;
}

// std::shared_ptr<UCXXRequest> UCXXEndpoint::createRequest(std::shared_ptr<ucxx_request_t> request)
// {
//   auto endpoint                       =
//   std::dynamic_pointer_cast<UCXXEndpoint>(shared_from_this()); auto req =
//   ucxx::createRequest(endpoint, _inflightRequests, request); std::weak_ptr<UCXXRequest> weak_req
//   = req; _inflightRequests->insert({req.get(), weak_req}); return req;
// }

std::shared_ptr<UCXXRequest> UCXXEndpoint::stream_send(void* buffer, size_t length)
{
  auto worker                         = UCXXEndpoint::getWorker(_parent);
  auto endpoint                       = std::dynamic_pointer_cast<UCXXEndpoint>(shared_from_this());
  auto request                        = createRequestStream(worker, endpoint, true, buffer, length);
  std::weak_ptr<UCXXRequest> weak_req = request;
  _inflightRequests->insert({request.get(), weak_req});
  return request;
}

std::shared_ptr<UCXXRequest> UCXXEndpoint::stream_recv(void* buffer, size_t length)
{
  auto worker   = UCXXEndpoint::getWorker(_parent);
  auto endpoint = std::dynamic_pointer_cast<UCXXEndpoint>(shared_from_this());
  auto request  = createRequestStream(worker, endpoint, false, buffer, length);
  std::weak_ptr<UCXXRequest> weak_req = request;
  _inflightRequests->insert({request.get(), weak_req});
  return request;
}

std::shared_ptr<UCXXRequest> UCXXEndpoint::tag_send(
  void* buffer,
  size_t length,
  ucp_tag_t tag,
  std::function<void(std::shared_ptr<void>)> callbackFunction,
  std::shared_ptr<void> callbackData)
{
  auto worker   = UCXXEndpoint::getWorker(_parent);
  auto endpoint = std::dynamic_pointer_cast<UCXXEndpoint>(shared_from_this());
  auto request =
    createRequestTag(worker, endpoint, true, buffer, length, tag, callbackFunction, callbackData);
  std::weak_ptr<UCXXRequest> weak_req = request;
  _inflightRequests->insert({request.get(), weak_req});
  return request;
}

std::shared_ptr<UCXXRequest> UCXXEndpoint::tag_recv(
  void* buffer,
  size_t length,
  ucp_tag_t tag,
  std::function<void(std::shared_ptr<void>)> callbackFunction,
  std::shared_ptr<void> callbackData)
{
  auto worker   = UCXXEndpoint::getWorker(_parent);
  auto endpoint = std::dynamic_pointer_cast<UCXXEndpoint>(shared_from_this());
  auto request =
    createRequestTag(worker, endpoint, false, buffer, length, tag, callbackFunction, callbackData);
  std::weak_ptr<UCXXRequest> weak_req = request;
  _inflightRequests->insert({request.get(), weak_req});
  return request;
}

}  // namespace ucxx
