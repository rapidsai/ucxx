/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint.h>
#include <ucxx/exception.h>
#include <ucxx/listener.h>
#include <ucxx/request_am.h>
#include <ucxx/request_data.h>
#include <ucxx/request_stream.h>
#include <ucxx/request_tag.h>
#include <ucxx/request_tag_multi.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils/callback_notifier.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>
#include <ucxx/worker.h>

namespace ucxx {

static std::shared_ptr<Worker> getWorker(std::shared_ptr<Component> workerOrListener)
{
  auto worker = std::dynamic_pointer_cast<Worker>(workerOrListener);
  if (worker == nullptr) {
    auto listener = std::dynamic_pointer_cast<Listener>(workerOrListener);
    if (listener == nullptr)
      throw std::invalid_argument(
        "Invalid object, it's not a shared_ptr to either ucxx::Worker nor ucxx::Listener");

    worker = std::dynamic_pointer_cast<Worker>(listener->getParent());
  }
  return worker;
}

Endpoint::Endpoint(std::shared_ptr<Component> workerOrListener,
                   ucp_ep_params_t* params,
                   bool endpointErrorHandling)
  : _endpointErrorHandling{endpointErrorHandling}
{
  auto worker = ::ucxx::getWorker(workerOrListener);

  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  setParent(workerOrListener);

  _callbackData = std::make_unique<ErrorCallbackData>(
    (ErrorCallbackData){.status = UCS_OK, .inflightRequests = _inflightRequests, .worker = worker});

  params->err_mode =
    (endpointErrorHandling ? UCP_ERR_HANDLING_MODE_PEER : UCP_ERR_HANDLING_MODE_NONE);
  params->err_handler.cb  = Endpoint::errorCallback;
  params->err_handler.arg = _callbackData.get();

  if (worker->isProgressThreadRunning()) {
    ucs_status_t status = UCS_INPROGRESS;
    utils::CallbackNotifier callbackNotifier{};
    auto worker = ::ucxx::getWorker(_parent);
    worker->registerGenericPre([this, &params, &callbackNotifier, &status]() {
      auto worker = ::ucxx::getWorker(_parent);
      status      = ucp_ep_create(worker->getHandle(), params, &_handle);
      callbackNotifier.set();
    });
    callbackNotifier.wait();
    utils::ucsErrorThrow(status);
  } else {
    utils::ucsErrorThrow(ucp_ep_create(worker->getHandle(), params, &_handle));
  }

  ucxx_trace("Endpoint created: %p, UCP handle: %p, parent: %p, endpointErrorHandling: %d",
             this,
             _handle,
             _parent.get(),
             endpointErrorHandling);
}

std::shared_ptr<Endpoint> createEndpointFromHostname(std::shared_ptr<Worker> worker,
                                                     std::string ipAddress,
                                                     uint16_t port,
                                                     bool endpointErrorHandling)
{
  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  ucp_ep_params_t params = {.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                          UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                          UCP_EP_PARAM_FIELD_ERR_HANDLER,
                            .flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER};
  auto info              = ucxx::utils::get_addrinfo(ipAddress.c_str(), port);

  params.sockaddr.addrlen = info->ai_addrlen;
  params.sockaddr.addr    = info->ai_addr;

  return std::shared_ptr<Endpoint>(new Endpoint(worker, &params, endpointErrorHandling));
}

std::shared_ptr<Endpoint> createEndpointFromConnRequest(std::shared_ptr<Listener> listener,
                                                        ucp_conn_request_h connRequest,
                                                        bool endpointErrorHandling)
{
  if (listener == nullptr || listener->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  ucp_ep_params_t params = {
    .field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_CONN_REQUEST |
                  UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE | UCP_EP_PARAM_FIELD_ERR_HANDLER,
    .flags        = UCP_EP_PARAMS_FLAGS_NO_LOOPBACK,
    .conn_request = connRequest};

  return std::shared_ptr<Endpoint>(new Endpoint(listener, &params, endpointErrorHandling));
}

std::shared_ptr<Endpoint> createEndpointFromWorkerAddress(std::shared_ptr<Worker> worker,
                                                          std::shared_ptr<Address> address,
                                                          bool endpointErrorHandling)
{
  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");
  if (address == nullptr || address->getHandle() == nullptr || address->getLength() == 0)
    throw ucxx::Error("Address not initialized");

  ucp_ep_params_t params = {.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                                          UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                          UCP_EP_PARAM_FIELD_ERR_HANDLER,
                            .address = address->getHandle()};

  return std::shared_ptr<Endpoint>(new Endpoint(worker, &params, endpointErrorHandling));
}

Endpoint::~Endpoint()
{
  close(10000000000 /* 10s */);
  ucxx_trace("Endpoint destroyed: %p, UCP handle: %p", this, _originalHandle);
}

void Endpoint::close(uint64_t period, uint64_t maxAttempts)
{
  if (_handle == nullptr) return;

  size_t canceled = cancelInflightRequests(3000000000 /* 3s */, 3);
  ucxx_debug("Endpoint %p canceled %lu requests", _handle, canceled);

  // Close the endpoint
  unsigned closeMode = UCP_EP_CLOSE_MODE_FORCE;
  if (_endpointErrorHandling && _callbackData->status != UCS_OK) {
    // We force close endpoint if endpoint error handling is enabled and
    // the endpoint status is not UCS_OK
    closeMode = UCP_EP_CLOSE_MODE_FORCE;
  }

  auto worker = ::ucxx::getWorker(_parent);
  ucs_status_ptr_t status;

  if (worker->isProgressThreadRunning()) {
    bool closeSuccess = false;
    for (uint64_t i = 0; i < maxAttempts && !closeSuccess; ++i) {
      utils::CallbackNotifier callbackNotifierPre{};
      worker->registerGenericPre([this, &callbackNotifierPre, &status, closeMode]() {
        status = ucp_ep_close_nb(_handle, closeMode);
        callbackNotifierPre.set();
      });
      if (!callbackNotifierPre.wait(period)) continue;

      while (UCS_PTR_IS_PTR(status)) {
        utils::CallbackNotifier callbackNotifierPost{};
        worker->registerGenericPost([this, &callbackNotifierPost, &status]() {
          ucs_status_t s = ucp_request_check_status(status);
          if (UCS_PTR_STATUS(s) != UCS_INPROGRESS) {
            ucp_request_free(status);
            _callbackData->status = UCS_PTR_STATUS(s);
            if (UCS_PTR_STATUS(status) != UCS_OK) {
              ucxx_error("Error while closing endpoint: %s",
                         ucs_status_string(UCS_PTR_STATUS(status)));
            }
          }

          callbackNotifierPost.set();
        });
        if (!callbackNotifierPost.wait(period)) continue;
      }

      closeSuccess = true;
    }

    if (!closeSuccess) {
      _callbackData->status = UCS_ERR_ENDPOINT_TIMEOUT;
      ucxx_error("All attempts to close timed out on endpoint: %p, UCP handle: %p", this, _handle);
    }
  } else {
    status = ucp_ep_close_nb(_handle, closeMode);
    if (UCS_PTR_IS_PTR(status)) {
      ucs_status_t s;
      while ((s = ucp_request_check_status(status)) == UCS_INPROGRESS)
        worker->progress();
      ucp_request_free(status);
      _callbackData->status = s;
    } else if (UCS_PTR_STATUS(status) != UCS_OK) {
      ucxx_error("Error while closing endpoint: %s", ucs_status_string(UCS_PTR_STATUS(status)));
    }
  }
  ucxx_trace("Endpoint closed: %p, UCP handle: %p", this, _handle);

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
    _callbackData->worker->scheduleRequestCancel(_inflightRequests->release());

  return request;
}

void Endpoint::removeInflightRequest(const Request* const request)
{
  _inflightRequests->remove(request);
}

size_t Endpoint::cancelInflightRequests(uint64_t period, uint64_t maxAttempts)
{
  auto worker     = ::ucxx::getWorker(this->_parent);
  size_t canceled = 0;

  if (std::this_thread::get_id() == worker->getProgressThreadId()) {
    canceled = _inflightRequests->cancelAll();
    for (uint64_t i = 0; i < maxAttempts && _inflightRequests->getCancelingCount() > 0; ++i)
      worker->progress();
  } else if (worker->isProgressThreadRunning()) {
    bool cancelSuccess = false;
    for (uint64_t i = 0; i < maxAttempts && !cancelSuccess; ++i) {
      utils::CallbackNotifier callbackNotifierPre{};
      worker->registerGenericPre([this, &callbackNotifierPre, &canceled]() {
        canceled += _inflightRequests->cancelAll();
        callbackNotifierPre.set();
      });
      if (!callbackNotifierPre.wait(period)) continue;

      utils::CallbackNotifier callbackNotifierPost{};
      worker->registerGenericPost([this, &callbackNotifierPost, &cancelSuccess]() {
        cancelSuccess = _inflightRequests->getCancelingCount() == 0;
        callbackNotifierPost.set();
      });
      if (!callbackNotifierPost.wait(period)) continue;
    }
    if (!cancelSuccess)
      ucxx_error("All attempts to cancel inflight requests failed on endpoint: %p, UCP handle: %p",
                 this,
                 _handle);
  } else {
    canceled = _inflightRequests->cancelAll();
  }

  return canceled;
}

std::shared_ptr<Request> Endpoint::amSend(void* buffer,
                                          size_t length,
                                          ucs_memory_type_t memoryType,
                                          const bool enablePythonFuture,
                                          RequestCallbackUserFunction callbackFunction,
                                          RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestAm(endpoint,
                                                 data::AmSend(buffer, length, memoryType),
                                                 enablePythonFuture,
                                                 callbackFunction,
                                                 callbackData));
}

std::shared_ptr<Request> Endpoint::amRecv(const bool enablePythonFuture,
                                          RequestCallbackUserFunction callbackFunction,
                                          RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestAm(
    endpoint, data::AmReceive(), enablePythonFuture, callbackFunction, callbackData));
}

std::shared_ptr<Request> Endpoint::streamSend(void* buffer,
                                              size_t length,
                                              const bool enablePythonFuture)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(
    createRequestStream(endpoint, data::StreamSend(buffer, length), enablePythonFuture));
}

std::shared_ptr<Request> Endpoint::streamRecv(void* buffer,
                                              size_t length,
                                              const bool enablePythonFuture)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(
    createRequestStream(endpoint, data::StreamReceive(buffer, length), enablePythonFuture));
}

std::shared_ptr<Request> Endpoint::tagSend(void* buffer,
                                           size_t length,
                                           Tag tag,
                                           const bool enablePythonFuture,
                                           RequestCallbackUserFunction callbackFunction,
                                           RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestTag(endpoint,
                                                  data::TagSend(buffer, length, tag),
                                                  enablePythonFuture,
                                                  callbackFunction,
                                                  callbackData));
}

std::shared_ptr<Request> Endpoint::tagRecv(void* buffer,
                                           size_t length,
                                           Tag tag,
                                           TagMask tagMask,
                                           const bool enablePythonFuture,
                                           RequestCallbackUserFunction callbackFunction,
                                           RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestTag(endpoint,
                                                  data::TagReceive(buffer, length, tag, tagMask),
                                                  enablePythonFuture,
                                                  callbackFunction,
                                                  callbackData));
}

std::shared_ptr<Request> Endpoint::tagMultiSend(const std::vector<void*>& buffer,
                                                const std::vector<size_t>& size,
                                                const std::vector<int>& isCUDA,
                                                const Tag tag,
                                                const bool enablePythonFuture)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestTagMulti(
    endpoint, data::TagMultiSend(buffer, size, isCUDA, tag), enablePythonFuture));
}

std::shared_ptr<Request> Endpoint::tagMultiRecv(const Tag tag,
                                                const TagMask tagMask,
                                                const bool enablePythonFuture)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(
    createRequestTagMulti(endpoint, data::TagMultiReceive(tag, tagMask), enablePythonFuture));
}

std::shared_ptr<Worker> Endpoint::getWorker() { return ::ucxx::getWorker(_parent); }

void Endpoint::errorCallback(void* arg, ucp_ep_h ep, ucs_status_t status)
{
  ErrorCallbackData* data = reinterpret_cast<ErrorCallbackData*>(arg);
  data->status            = status;
  data->worker->scheduleRequestCancel(data->inflightRequests->release());
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
