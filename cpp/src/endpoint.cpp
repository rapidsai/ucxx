/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <ucp/api/ucp_compat.h>
#include <ucs/type/status.h>
#include <utility>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint.h>
#include <ucxx/exception.h>
#include <ucxx/listener.h>
#include <ucxx/remote_key.h>
#include <ucxx/request_am.h>
#include <ucxx/request_data.h>
#include <ucxx/request_endpoint_close.h>
#include <ucxx/request_flush.h>
#include <ucxx/request_mem.h>
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

void endpointErrorCallback(void* arg, ucp_ep_h ep, ucs_status_t status)
{
  auto endpoint = static_cast<Endpoint*>(arg);

  // Unable to cast to `Endpoint*`: invalid `arg`.
  if (endpoint == nullptr) {
    ucxx_error("ucxx::Endpoint::%s, UCP handle: %p, error callback called with status %d: %s",
               __func__,
               ep,
               status,
               ucs_status_string(status));
    return;
  }

  // Endpoint is already closing.
  if (endpoint->_closing.exchange(true)) return;

  endpoint->_status = status;
  auto worker       = ::ucxx::getWorker(endpoint->_parent);
  worker->scheduleRequestCancel(endpoint->_inflightRequests->release());
  {
    std::lock_guard<std::mutex> lock(endpoint->_mutex);
    if (endpoint->_closeCallback) {
      ucxx_debug("ucxx::Endpoint::%s: %p, UCP handle: %p, calling user close callback",
                 __func__,
                 endpoint,
                 ep);
      endpoint->_closeCallback(status, endpoint->_closeCallbackArg);
      endpoint->_closeCallback    = nullptr;
      endpoint->_closeCallbackArg = nullptr;
    }
  }

  // Connection reset and timeout often represent just a normal remote
  // endpoint disconnect, log only in debug mode.
  if (status == UCS_ERR_CONNECTION_RESET || status == UCS_ERR_ENDPOINT_TIMEOUT)
    ucxx_debug("ucxx::Endpoint::%s: %p, UCP handle: %p, error callback called with status %d: %s",
               __func__,
               endpoint,
               ep,
               status,
               ucs_status_string(status));
  else
    ucxx_error("ucxx::Endpoint::%s: %p, UCP handle: %p, error callback called with status %d: %s",
               __func__,
               endpoint,
               ep,
               status,
               ucs_status_string(status));
}

Endpoint::Endpoint(std::shared_ptr<Component> workerOrListener, bool endpointErrorHandling)
  : _endpointErrorHandling{endpointErrorHandling}
{
  auto worker = ::ucxx::getWorker(workerOrListener);

  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  setParent(workerOrListener);
}

void Endpoint::create(ucp_ep_params_t* params)
{
  auto worker = ::ucxx::getWorker(_parent);

  if (_endpointErrorHandling) {
    params->err_mode        = UCP_ERR_HANDLING_MODE_PEER;
    params->err_handler.cb  = endpointErrorCallback;
    params->err_handler.arg = this;
  } else {
    params->err_mode        = UCP_ERR_HANDLING_MODE_NONE;
    params->err_handler.cb  = nullptr;
    params->err_handler.arg = nullptr;
  }

  if (worker->isProgressThreadRunning()) {
    ucs_status_t status = UCS_INPROGRESS;

    size_t maxAttempts = 3;
    for (uint64_t i = 0; i < maxAttempts; ++i) {
      if (worker->registerGenericPre(
            [this, &worker, &params, &status]() {
              status = ucp_ep_create(worker->getHandle(), params, &_handle);
            },
            3000000000 /* 3s */))
        break;

      if (i == maxAttempts - 1) {
        status = UCS_ERR_TIMED_OUT;
        ucxx_error("Timeout waiting for ucp_ep_create, all attempts failed");
      } else {
        ucxx_warn("Timeout waiting for ucp_ep_create, retrying");
      }
    }
    utils::ucsErrorThrow(status);
  } else {
    utils::ucsErrorThrow(ucp_ep_create(worker->getHandle(), params, &_handle));
  }

  ucxx_trace("ucxx::Endpoint created: %p, UCP handle: %p, parent: %p, endpointErrorHandling: %d",
             this,
             _handle,
             _parent.get(),
             _endpointErrorHandling);
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

  auto ep = std::shared_ptr<Endpoint>(new Endpoint(worker, endpointErrorHandling));
  ep->create(&params);
  return ep;
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

  auto ep = std::shared_ptr<Endpoint>(new Endpoint(listener, endpointErrorHandling));
  ep->create(&params);
  return ep;
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

  auto ep = std::shared_ptr<Endpoint>(new Endpoint(worker, endpointErrorHandling));
  ep->create(&params);
  return ep;
}

Endpoint::~Endpoint()
{
  closeBlocking(10000000000 /* 10s */);
  ucxx_trace("ucxx::Endpoint destroyed: %p, UCP handle: %p", this, _originalHandle);
}

std::shared_ptr<Request> Endpoint::close(const bool enablePythonFuture,
                                         EndpointCloseCallbackUserFunction callbackFunction,
                                         EndpointCloseCallbackUserData callbackData)
{
  if (_closing.exchange(true) || _handle == nullptr) return nullptr;

  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  bool force    = _endpointErrorHandling;

  auto combineCallbacksFunction = [this, &callbackFunction, &callbackData](
                                    ucs_status_t status, EndpointCloseCallbackUserData unused) {
    _status = status;
    if (callbackFunction) callbackFunction(status, callbackData);
    {
      std::lock_guard<std::mutex> lock(_mutex);
      if (_closeCallback) {
        _closeCallback(status, _closeCallbackArg);
        _closeCallback    = nullptr;
        _closeCallbackArg = nullptr;
      }
    }
  };

  return registerInflightRequest(createRequestEndpointClose(
    endpoint, data::EndpointClose(force), enablePythonFuture, combineCallbacksFunction, nullptr));
}

void Endpoint::closeBlocking(uint64_t period, uint64_t maxAttempts)
{
  if (_closing.exchange(true) || _handle == nullptr) return;

  size_t canceled = cancelInflightRequestsBlocking(3000000000 /* 3s */, 3);
  ucxx_debug("ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, canceled %lu requests",
             __func__,
             this,
             _handle,
             canceled);

  ucp_request_param_t param{};
  if (_endpointErrorHandling)
    param = {.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS, .flags = UCP_EP_CLOSE_FLAG_FORCE};

  auto worker             = ::ucxx::getWorker(_parent);
  ucs_status_ptr_t status = nullptr;

  if (worker->isProgressThreadRunning()) {
    bool closeSuccess = false;
    bool submitted    = false;
    for (uint64_t i = 0; i < maxAttempts && !closeSuccess; ++i) {
      if (!submitted) {
        if (!worker->registerGenericPre(
              [this, &status, &param]() { status = ucp_ep_close_nbx(_handle, &param); }, period))
          continue;
        submitted = true;
      }

      if (_status == UCS_INPROGRESS) {
        if (!worker->registerGenericPost(
              [this, &status]() {
                if (UCS_PTR_IS_PTR(status)) {
                  ucs_status_t s;
                  if ((s = ucp_request_check_status(status)) != UCS_INPROGRESS) { _status = s; }
                } else if (UCS_PTR_STATUS(status) != UCS_OK) {
                  ucxx_error(
                    "ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, Error while closing "
                    "endpoint: %s",
                    __func__,
                    this,
                    _handle,
                    ucs_status_string(UCS_PTR_STATUS(status)));
                }
              },
              period))
          continue;
      }

      closeSuccess = true;
    }

    if (!closeSuccess) {
      _status = UCS_ERR_ENDPOINT_TIMEOUT;
      ucxx_debug(
        "ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, all attempts to close timed out",
        __func__,
        this,
        _handle);
    }
  } else {
    status = ucp_ep_close_nbx(_handle, &param);
    if (UCS_PTR_IS_PTR(status)) {
      ucs_status_t s;
      while ((s = ucp_request_check_status(status)) == UCS_INPROGRESS)
        worker->progress();
      _status = s;
    } else if (UCS_PTR_STATUS(status) != UCS_OK) {
      ucxx_error(
        "ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, Error while closing endpoint: %s",
        __func__,
        this,
        _handle,
        ucs_status_string(UCS_PTR_STATUS(status)));
    }
  }
  ucxx_trace("ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, closed", __func__, this, _handle);

  if (UCS_PTR_IS_PTR(status)) ucp_request_free(status);

  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_closeCallback) {
      ucxx_debug("ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, calling user close callback",
                 __func__,
                 this,
                 _handle);
      _closeCallback(_status, _closeCallbackArg);
      _closeCallback    = nullptr;
      _closeCallbackArg = nullptr;
    }
  }

  std::swap(_handle, _originalHandle);
}

ucp_ep_h Endpoint::getHandle() { return _handle; }

bool Endpoint::isAlive() const
{
  if (!_endpointErrorHandling) return true;

  return _status == UCS_INPROGRESS;
}

void Endpoint::raiseOnError()
{
  ucs_status_t status = _status;

  if (status == UCS_OK || status == UCS_INPROGRESS || !_endpointErrorHandling) return;

  std::string statusString{ucs_status_string(status)};
  std::stringstream errorMsgStream;
  errorMsgStream << "Endpoint " << std::hex << _handle << " error: " << statusString;

  utils::ucsErrorThrow(status, errorMsgStream.str());
}

void Endpoint::setCloseCallback(EndpointCloseCallbackUserFunction closeCallback,
                                EndpointCloseCallbackUserData closeCallbackArg)
{
  std::lock_guard<std::mutex> lock(_mutex);

  if (_closing.load() && closeCallback != nullptr && closeCallbackArg != nullptr)
    throw std::runtime_error("Endpoint is closing or has already closed.");

  _closeCallback    = closeCallback;
  _closeCallbackArg = closeCallbackArg;
}

std::shared_ptr<Request> Endpoint::registerInflightRequest(std::shared_ptr<Request> request)
{
  if (!request->isCompleted()) _inflightRequests->insert(request);

  /**
   * If the endpoint closed or errored while the request was being submitted, the error
   * handler may have been called already and we need to register any new requests for
   * cancelation, including the present one.
   */
  if (_status != UCS_INPROGRESS) {
    auto worker = ::ucxx::getWorker(_parent);
    worker->scheduleRequestCancel(_inflightRequests->release());
  }

  return request;
}

void Endpoint::removeInflightRequest(const Request* const request)
{
  _inflightRequests->remove(request);
}

size_t Endpoint::cancelInflightRequests() { return _inflightRequests->cancelAll(); }

size_t Endpoint::cancelInflightRequestsBlocking(uint64_t period, uint64_t maxAttempts)
{
  auto worker     = ::ucxx::getWorker(_parent);
  size_t canceled = 0;

  if (std::this_thread::get_id() == worker->getProgressThreadId()) {
    canceled = _inflightRequests->cancelAll();
    for (uint64_t i = 0; i < maxAttempts && _inflightRequests->getCancelingSize() > 0; ++i)
      worker->progress();
  } else if (worker->isProgressThreadRunning()) {
    bool cancelSuccess = false;
    for (uint64_t i = 0; i < maxAttempts && !cancelSuccess; ++i) {
      if (!worker->registerGenericPre(
            [this, &canceled]() { canceled += _inflightRequests->cancelAll(); }, period))
        continue;

      if (!worker->registerGenericPost(
            [this, &cancelSuccess]() {
              cancelSuccess = _inflightRequests->getCancelingSize() == 0;
            },
            period))
        continue;
    }
    if (!cancelSuccess)
      ucxx_debug(
        "ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, all attempts to "
        "cancel inflight requests failed",
        __func__,
        this,
        _handle);
  } else {
    canceled = _inflightRequests->cancelAll();
  }

  return canceled;
}

size_t Endpoint::getCancelingSize() const { return _inflightRequests->getCancelingSize(); }

std::shared_ptr<Request> Endpoint::amSend(
  void* buffer,
  const size_t length,
  const ucs_memory_type_t memoryType,
  const std::optional<AmReceiverCallbackInfo> receiverCallbackInfo,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(
    createRequestAm(endpoint,
                    data::AmSend(buffer, length, memoryType, receiverCallbackInfo),
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

std::shared_ptr<Request> Endpoint::memGet(void* buffer,
                                          size_t length,
                                          uint64_t remoteAddr,
                                          ucp_rkey_h rkey,
                                          const bool enablePythonFuture,
                                          RequestCallbackUserFunction callbackFunction,
                                          RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestMem(endpoint,
                                                  data::MemGet(buffer, length, remoteAddr, rkey),
                                                  enablePythonFuture,
                                                  callbackFunction,
                                                  callbackData));
}

std::shared_ptr<Request> Endpoint::memGet(void* buffer,
                                          size_t length,
                                          std::shared_ptr<RemoteKey> remoteKey,
                                          uint64_t remoteAddressOffset,
                                          const bool enablePythonFuture,
                                          RequestCallbackUserFunction callbackFunction,
                                          RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestMem(
    endpoint,
    data::MemGet(
      buffer, length, remoteKey->getBaseAddress() + remoteAddressOffset, remoteKey->getHandle()),
    enablePythonFuture,
    callbackFunction,
    callbackData));
}

std::shared_ptr<Request> Endpoint::memPut(void* buffer,
                                          size_t length,
                                          uint64_t remoteAddr,
                                          ucp_rkey_h rkey,
                                          const bool enablePythonFuture,
                                          RequestCallbackUserFunction callbackFunction,
                                          RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestMem(endpoint,
                                                  data::MemPut(buffer, length, remoteAddr, rkey),
                                                  enablePythonFuture,
                                                  callbackFunction,
                                                  callbackData));
}

std::shared_ptr<Request> Endpoint::memPut(void* buffer,
                                          size_t length,
                                          std::shared_ptr<RemoteKey> remoteKey,
                                          uint64_t remoteAddressOffset,
                                          const bool enablePythonFuture,
                                          RequestCallbackUserFunction callbackFunction,
                                          RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestMem(
    endpoint,
    data::MemPut(
      buffer, length, remoteKey->getBaseAddress() + remoteAddressOffset, remoteKey->getHandle()),
    enablePythonFuture,
    callbackFunction,
    callbackData));
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

std::shared_ptr<Request> Endpoint::flush(const bool enablePythonFuture,
                                         RequestCallbackUserFunction callbackFunction,
                                         RequestCallbackUserData callbackData)
{
  auto endpoint = std::dynamic_pointer_cast<Endpoint>(shared_from_this());
  return registerInflightRequest(createRequestFlush(
    endpoint, data::Flush(), enablePythonFuture, callbackFunction, callbackData));
}

std::shared_ptr<Worker> Endpoint::getWorker() { return ::ucxx::getWorker(_parent); }

}  // namespace ucxx
