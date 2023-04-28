/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint.h>
#include <ucxx/exception.h>
#include <ucxx/listener.h>
#include <ucxx/request_stream.h>
#include <ucxx/request_tag.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>
#include <ucxx/worker.h>

namespace ucxx {

void EpParamsDeleter::operator()(ucp_ep_params_t* ptr)
{
  if (ptr != nullptr && ptr->field_mask & UCP_EP_PARAM_FIELD_SOCK_ADDR)
    ucxx::utils::sockaddr_free(&ptr->sockaddr);
}

Endpoint::Endpoint(std::shared_ptr<Component> workerOrListener,
                   std::unique_ptr<ucp_ep_params_t, EpParamsDeleter> params,
                   bool endpointErrorHandling)
  : _endpointErrorHandling{endpointErrorHandling}
{
  auto worker = Endpoint::getWorker(workerOrListener);

  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  setParent(workerOrListener);

  _callbackData = std::make_unique<ErrorCallbackData>(
    (ErrorCallbackData){.status = UCS_OK, .inflightRequests = _inflightRequests, .worker = worker});

  params->err_mode =
    (endpointErrorHandling ? UCP_ERR_HANDLING_MODE_PEER : UCP_ERR_HANDLING_MODE_NONE);
  params->err_handler.cb  = Endpoint::errorCallback;
  params->err_handler.arg = _callbackData.get();

  utils::ucsErrorThrow(ucp_ep_create(worker->getHandle(), params.get(), &_handle));
  ucxx_trace("Endpoint created: %p", _handle);
}

std::shared_ptr<Endpoint> createEndpointFromHostname(std::shared_ptr<Worker> worker,
                                                     std::string ipAddress,
                                                     uint16_t port,
                                                     bool endpointErrorHandling)
{
  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  auto params = std::unique_ptr<ucp_ep_params_t, EpParamsDeleter>(new ucp_ep_params_t);

  struct hostent* hostname = gethostbyname(ipAddress.c_str());
  if (hostname == nullptr) throw ucxx::Error(std::string("Invalid IP address or hostname"));

  params->field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                       UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE | UCP_EP_PARAM_FIELD_ERR_HANDLER;
  params->flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
  if (ucxx::utils::sockaddr_set(&params->sockaddr, hostname->h_name, port)) throw std::bad_alloc();

  return std::shared_ptr<Endpoint>(new Endpoint(worker, std::move(params), endpointErrorHandling));
}

std::shared_ptr<Endpoint> createEndpointFromConnRequest(std::shared_ptr<Listener> listener,
                                                        ucp_conn_request_h connRequest,
                                                        bool endpointErrorHandling)
{
  if (listener == nullptr || listener->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  auto params        = std::unique_ptr<ucp_ep_params_t, EpParamsDeleter>(new ucp_ep_params_t);
  params->field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_CONN_REQUEST |
                       UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE | UCP_EP_PARAM_FIELD_ERR_HANDLER;
  params->flags        = UCP_EP_PARAMS_FLAGS_NO_LOOPBACK;
  params->conn_request = connRequest;

  return std::shared_ptr<Endpoint>(
    new Endpoint(listener, std::move(params), endpointErrorHandling));
}

std::shared_ptr<Endpoint> createEndpointFromWorkerAddress(std::shared_ptr<Worker> worker,
                                                          std::shared_ptr<Address> address,
                                                          bool endpointErrorHandling)
{
  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");
  if (address == nullptr || address->getHandle() == nullptr || address->getLength() == 0)
    throw ucxx::Error("Address not initialized");

  auto params        = std::unique_ptr<ucp_ep_params_t, EpParamsDeleter>(new ucp_ep_params_t);
  params->field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                       UCP_EP_PARAM_FIELD_ERR_HANDLER;
  params->address = address->getHandle();

  return std::shared_ptr<Endpoint>(new Endpoint(worker, std::move(params), endpointErrorHandling));
}

Endpoint::~Endpoint()
{
  close();
  ucxx_trace("Endpoint destroyed: %p", _originalHandle);
}

void Endpoint::close()
{
  if (_handle == nullptr) return;

  size_t canceled = cancelInflightRequests();
  ucxx_debug("Endpoint %p canceled %lu requests", _handle, canceled);

  // Close the endpoint
  unsigned closeMode = UCP_EP_CLOSE_MODE_FORCE;
  if (_endpointErrorHandling && _callbackData->status != UCS_OK) {
    // We force close endpoint if endpoint error handling is enabled and
    // the endpoint status is not UCS_OK
    closeMode = UCP_EP_CLOSE_MODE_FORCE;
  }
  ucs_status_ptr_t status = ucp_ep_close_nb(_handle, closeMode);
  if (UCS_PTR_IS_PTR(status)) {
    auto worker = Endpoint::getWorker(_parent);
    while (ucp_request_check_status(status) == UCS_INPROGRESS)
      worker->progress();
    ucp_request_free(status);
  } else if (UCS_PTR_STATUS(status) != UCS_OK) {
    ucxx_error("Error while closing endpoint: %s", ucs_status_string(UCS_PTR_STATUS(status)));
  }
  ucxx_trace("Endpoint closed: %p", _handle);

  if (_callbackData->closeCallback) {
    ucxx_debug("Calling user callback for endpoint %p", _handle);
    _callbackData->closeCallback(_callbackData->closeCallbackArg);
    _callbackData->closeCallback    = nullptr;
    _callbackData->closeCallbackArg = nullptr;
  }

  std::swap(_handle, _originalHandle);
}

ucp_ep_h Endpoint::getHandle() { return _handle; }

bool Endpoint::isAlive() const
{
  if (!_endpointErrorHandling) return true;

  return _callbackData->status == UCS_OK;
}

void Endpoint::raiseOnError()
{
  ucs_status_t status = _callbackData->status;

  if (status == UCS_OK || !_endpointErrorHandling) return;

  std::string statusString{ucs_status_string(status)};
  std::stringstream errorMsgStream;
  errorMsgStream << "Endpoint " << std::hex << _handle << " error: " << statusString;

  utils::ucsErrorThrow(status, errorMsgStream.str());
}

void Endpoint::setCloseCallback(std::function<void(void*)> closeCallback, void* closeCallbackArg)
{
  _callbackData->closeCallback    = closeCallback;
  _callbackData->closeCallbackArg = closeCallbackArg;
}

std::shared_ptr<Request> Endpoint::registerInflightRequest(std::shared_ptr<Request> request)
{
  if (!request->isCompleted()) _inflightRequests->insert(request);

  /**
   * If the endpoint errored while the request was being submitted, the error
   * handler may have been called already and we need to register any new requests
   * for cancelation, including the present one.
   */
  if (_callbackData->status != UCS_OK)
    _callbackData->worker->scheduleRequestCancel(_inflightRequests);

  return request;
}

void Endpoint::removeInflightRequest(const Request* const request)
{
  _inflightRequests->remove(request);
}

size_t Endpoint::cancelInflightRequests() { return _inflightRequests->cancelAll(); }

std::shared_ptr<Request> Endpoint::streamSend(void* buffer,
                                              size_t length,
                                              const bool enablePythonFuture)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(
    createRequestStream(endpoint, true, buffer, length, enablePythonFuture));
}

std::shared_ptr<Request> Endpoint::streamRecv(void* buffer,
                                              size_t length,
                                              const bool enablePythonFuture)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(
    createRequestStream(endpoint, false, buffer, length, enablePythonFuture));
}

std::shared_ptr<Request> Endpoint::tagSend(void* buffer,
                                           size_t length,
                                           ucp_tag_t tag,
                                           const bool enablePythonFuture,
                                           RequestCallbackUserFunction callbackFunction,
                                           RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestTag(
    endpoint, true, buffer, length, tag, enablePythonFuture, callbackFunction, callbackData));
}

std::shared_ptr<Request> Endpoint::tagRecv(void* buffer,
                                           size_t length,
                                           ucp_tag_t tag,
                                           const bool enablePythonFuture,
                                           RequestCallbackUserFunction callbackFunction,
                                           RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestTag(
    endpoint, false, buffer, length, tag, enablePythonFuture, callbackFunction, callbackData));
}

std::shared_ptr<RequestTagMulti> Endpoint::tagMultiSend(const std::vector<void*>& buffer,
                                                        const std::vector<size_t>& size,
                                                        const std::vector<int>& isCUDA,
                                                        const ucp_tag_t tag,
                                                        const bool enablePythonFuture)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return createRequestTagMultiSend(endpoint, buffer, size, isCUDA, tag, enablePythonFuture);
}

std::shared_ptr<RequestTagMulti> Endpoint::tagMultiRecv(const ucp_tag_t tag,
                                                        const bool enablePythonFuture)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return createRequestTagMultiRecv(endpoint, tag, enablePythonFuture);
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
  ErrorCallbackData* data = reinterpret_cast<ErrorCallbackData*>(arg);
  data->status            = status;
  data->worker->scheduleRequestCancel(data->inflightRequests);
  if (data->closeCallback) {
    ucxx_debug("Calling user callback for endpoint %p", ep);
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
