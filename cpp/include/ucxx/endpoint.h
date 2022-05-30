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
#include <ucxx/listener.h>
#include <ucxx/request.h>
#include <ucxx/sockaddr_utils.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils.h>
#include <ucxx/worker.h>

namespace ucxx {

struct EpParamsDeleter {
  void operator()(ucp_ep_params_t* ptr);
};

typedef struct error_callback_data {
  ucs_status_t status;
  inflight_requests_t inflightRequests;
  std::function<void(void*)> closeCallback;
  void* closeCallbackArg;
  std::shared_ptr<Worker> worker;
} error_callback_data_t;

class Endpoint : public Component {
 private:
  ucp_ep_h _handle{nullptr};
  bool _endpoint_error_handling{true};
  std::unique_ptr<error_callback_data_t> _callbackData{nullptr};
  inflight_requests_t _inflightRequests{std::make_shared<inflight_request_map_t>()};

  Endpoint(std::shared_ptr<Component> worker_or_listener,
           std::unique_ptr<ucp_ep_params_t, EpParamsDeleter> params,
           bool endpoint_error_handling);

 public:
  Endpoint()                = delete;
  Endpoint(const Endpoint&) = delete;
  Endpoint& operator=(Endpoint const&) = delete;
  Endpoint(Endpoint&& o)               = delete;
  Endpoint& operator=(Endpoint&& o) = delete;

  ~Endpoint();

  friend std::shared_ptr<Endpoint> createEndpointFromHostname(std::shared_ptr<Worker> worker,
                                                              std::string ip_address,
                                                              uint16_t port,
                                                              bool endpoint_error_handling)
  {
    if (worker == nullptr || worker->get_handle() == nullptr)
      throw ucxx::Error("Worker not initialized");

    auto params = std::unique_ptr<ucp_ep_params_t, EpParamsDeleter>(new ucp_ep_params_t);

    struct hostent* hostname = gethostbyname(ip_address.c_str());
    if (hostname == nullptr) throw ucxx::Error(std::string("Invalid IP address or hostname"));

    params->field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE | UCP_EP_PARAM_FIELD_ERR_HANDLER;
    params->flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    if (sockaddr_utils_set(&params->sockaddr, hostname->h_name, port)) throw std::bad_alloc();

    return std::shared_ptr<Endpoint>(
      new Endpoint(worker, std::move(params), endpoint_error_handling));
  }

  friend std::shared_ptr<Endpoint> createEndpointFromConnRequest(std::shared_ptr<Listener> listener,
                                                                 ucp_conn_request_h conn_request,
                                                                 bool endpoint_error_handling)
  {
    if (listener == nullptr || listener->get_handle() == nullptr)
      throw ucxx::Error("Worker not initialized");

    auto params        = std::unique_ptr<ucp_ep_params_t, EpParamsDeleter>(new ucp_ep_params_t);
    params->field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_CONN_REQUEST |
                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE | UCP_EP_PARAM_FIELD_ERR_HANDLER;
    params->flags        = UCP_EP_PARAMS_FLAGS_NO_LOOPBACK;
    params->conn_request = conn_request;

    return std::shared_ptr<Endpoint>(
      new Endpoint(listener, std::move(params), endpoint_error_handling));
  }

  friend std::shared_ptr<Endpoint> createEndpointFromWorkerAddress(std::shared_ptr<Worker> worker,
                                                                   std::shared_ptr<Address> address,
                                                                   bool endpoint_error_handling)
  {
    if (worker == nullptr || worker->get_handle() == nullptr)
      throw ucxx::Error("Worker not initialized");
    if (address == nullptr || address->getHandle() == nullptr || address->getLength() == 0)
      throw ucxx::Error("Address not initialized");

    auto params        = std::unique_ptr<ucp_ep_params_t, EpParamsDeleter>(new ucp_ep_params_t);
    params->field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                         UCP_EP_PARAM_FIELD_ERR_HANDLER;
    params->address = address->getHandle();

    return std::shared_ptr<Endpoint>(
      new Endpoint(worker, std::move(params), endpoint_error_handling));
  }

  ucp_ep_h getHandle();

  bool isAlive() const;

  void raiseOnError();

  void registerInflightRequest(std::shared_ptr<Request> request);

  // void removeInflightRequest(std::shared_ptr<Request> request);
  void removeInflightRequest(Request* request);

  void setCloseCallback(std::function<void(void*)> closeCallback, void* closeCallbackArg);

  std::shared_ptr<Request> stream_send(void* buffer, size_t length);

  std::shared_ptr<Request> stream_recv(void* buffer, size_t length);

  std::shared_ptr<Request> tag_send(
    void* buffer,
    size_t length,
    ucp_tag_t tag,
    const bool enablePythonFuture                               = true,
    std::function<void(std::shared_ptr<void>)> callbackFunction = nullptr,
    std::shared_ptr<void> callbackData                          = nullptr);

  std::shared_ptr<Request> tag_recv(
    void* buffer,
    size_t length,
    ucp_tag_t tag,
    const bool enablePythonFuture                               = true,
    std::function<void(std::shared_ptr<void>)> callbackFunction = nullptr,
    std::shared_ptr<void> callbackData                          = nullptr);

  static std::shared_ptr<Worker> getWorker(std::shared_ptr<Component> workerOrListener);

  static void errorCallback(void* arg, ucp_ep_h ep, ucs_status_t status);
};

}  // namespace ucxx
