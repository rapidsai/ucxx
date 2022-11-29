/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <netdb.h>

#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/address.h>
#include <ucxx/component.h>
#include <ucxx/exception.h>
#include <ucxx/inflight_requests.h>
#include <ucxx/listener.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/worker.h>

namespace ucxx {

struct EpParamsDeleter {
  void operator()(ucp_ep_params_t* ptr);
};

struct ErrorCallbackData {
  ucs_status_t status;
  std::shared_ptr<InflightRequests> inflightRequests;
  std::function<void(void*)> closeCallback;
  void* closeCallbackArg;
  std::shared_ptr<Worker> worker;
};

class Endpoint : public Component {
 private:
  ucp_ep_h _handle{nullptr};
  bool _endpointErrorHandling{true};
  std::unique_ptr<ErrorCallbackData> _callbackData{nullptr};
  std::shared_ptr<InflightRequests> _inflightRequests{std::make_shared<InflightRequests>()};

  Endpoint(std::shared_ptr<Component> workerOrListener,
           std::unique_ptr<ucp_ep_params_t, EpParamsDeleter> params,
           bool endpointErrorHandling);

  void registerInflightRequest(std::shared_ptr<Request> request);

 public:
  Endpoint()                = delete;
  Endpoint(const Endpoint&) = delete;
  Endpoint& operator=(Endpoint const&) = delete;
  Endpoint(Endpoint&& o)               = delete;
  Endpoint& operator=(Endpoint&& o) = delete;

  ~Endpoint();

  friend std::shared_ptr<Endpoint> createEndpointFromHostname(std::shared_ptr<Worker> worker,
                                                              std::string ipAddress,
                                                              uint16_t port,
                                                              bool endpointErrorHandling);

  friend std::shared_ptr<Endpoint> createEndpointFromConnRequest(std::shared_ptr<Listener> listener,
                                                                 ucp_conn_request_h connRequest,
                                                                 bool endpointErrorHandling);

  friend std::shared_ptr<Endpoint> createEndpointFromWorkerAddress(std::shared_ptr<Worker> worker,
                                                                   std::shared_ptr<Address> address,
                                                                   bool endpointErrorHandling);

  ucp_ep_h getHandle();

  bool isAlive() const;

  void raiseOnError();

  void removeInflightRequest(const Request* const request);

  size_t cancelInflightRequests();

  void setCloseCallback(std::function<void(void*)> closeCallback, void* closeCallbackArg);

  std::shared_ptr<Request> streamSend(void* buffer, size_t length, const bool enablePythonFuture);

  std::shared_ptr<Request> streamRecv(void* buffer, size_t length, const bool enablePythonFuture);

  std::shared_ptr<Request> tagSend(
    void* buffer,
    size_t length,
    ucp_tag_t tag,
    const bool enablePythonFuture                               = false,
    std::function<void(std::shared_ptr<void>)> callbackFunction = nullptr,
    std::shared_ptr<void> callbackData                          = nullptr);

  std::shared_ptr<Request> tagRecv(
    void* buffer,
    size_t length,
    ucp_tag_t tag,
    const bool enablePythonFuture                               = false,
    std::function<void(std::shared_ptr<void>)> callbackFunction = nullptr,
    std::shared_ptr<void> callbackData                          = nullptr);

  std::shared_ptr<RequestTagMulti> tagMultiSend(std::vector<void*>& buffer,
                                                std::vector<size_t>& size,
                                                std::vector<int>& isCUDA,
                                                const ucp_tag_t tag,
                                                const bool enablePythonFuture);

  std::shared_ptr<RequestTagMulti> tagMultiRecv(const ucp_tag_t tag, const bool enablePythonFuture);

  static std::shared_ptr<Worker> getWorker(std::shared_ptr<Component> workerOrListener);

  static void errorCallback(void* arg, ucp_ep_h ep, ucs_status_t status);

  void close();
};

}  // namespace ucxx
