/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <arpa/inet.h>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ucp/api/ucp.h>

#if UCXX_ENABLE_CCCL
#include <cuda_runtime_api.h>
#endif

namespace {

constexpr size_t NumMessages{4};
constexpr size_t MessageSize{1 << 20};
constexpr ucp_tag_t Tag{0};

struct Completion {
  bool completed{false};
  ucs_status_t status{UCS_INPROGRESS};
};

struct EndpointError {
  ucs_status_t status{UCS_OK};
};

struct ListenerState {
  ucp_worker_h worker{nullptr};
  ucp_ep_h endpoint{nullptr};
  EndpointError endpointError{};
  ucs_status_t status{UCS_OK};
};

struct ReceiveState {
  ucp_worker_h worker{nullptr};
  std::vector<void*> buffers{};
  Completion header{};
  std::vector<Completion> frames{NumMessages};
  ucs_status_t postStatus{UCS_OK};
};

void sendCallback(void* request, ucs_status_t status, void* userData)
{
  auto* completion      = static_cast<Completion*>(userData);
  completion->status    = status;
  completion->completed = true;
  ucp_request_free(request);
}

void recvCallback(void* request,
                  ucs_status_t status,
                  const ucp_tag_recv_info_t* /* tagInfo */,
                  void* userData)
{
  auto* completion      = static_cast<Completion*>(userData);
  completion->status    = status;
  completion->completed = true;
  ucp_request_free(request);
}

void headerRecvCallback(void* request,
                        ucs_status_t status,
                        const ucp_tag_recv_info_t* /* tagInfo */,
                        void* userData)
{
  auto* receiveState             = static_cast<ReceiveState*>(userData);
  receiveState->header.status    = status;
  receiveState->header.completed = true;
  ucp_request_free(request);

  if (status != UCS_OK) return;

  ucp_request_param_t recvParams{};
  recvParams.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  recvParams.cb.recv      = recvCallback;
  for (size_t i = 0; i < NumMessages; ++i) {
    recvParams.user_data = &receiveState->frames[i];
    const auto request   = ucp_tag_recv_nbx(
      receiveState->worker, receiveState->buffers[i], MessageSize, Tag, UINT64_MAX, &recvParams);
    if (UCS_PTR_IS_ERR(request)) {
      receiveState->postStatus = UCS_PTR_STATUS(request);
      return;
    }
    if (request == nullptr) {
      receiveState->frames[i].completed = true;
      receiveState->frames[i].status    = UCS_OK;
    }
  }
}

void endpointErrorCallback(void* arg, ucp_ep_h /* endpoint */, ucs_status_t status)
{
  static_cast<EndpointError*>(arg)->status = status;
}

void listenerCallback(ucp_conn_request_h connRequest, void* arg)
{
  auto* state = static_cast<ListenerState*>(arg);

  ucp_ep_params_t params{};
  params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                      UCP_EP_PARAM_FIELD_ERR_HANDLER;
  params.conn_request    = connRequest;
  params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
  params.err_handler.cb  = endpointErrorCallback;
  params.err_handler.arg = &state->endpointError;
  state->status          = ucp_ep_create(state->worker, &params, &state->endpoint);
}

class RawUcxTagCloseTest : public ::testing::Test {
 protected:
  ucp_context_h _context{nullptr};
  ucp_worker_h _clientWorker{nullptr};
  ucp_worker_h _serverWorker{nullptr};
  ucp_listener_h _listener{nullptr};
  ucp_ep_h _clientEndpoint{nullptr};
  ListenerState _listenerState{};
  EndpointError _clientEndpointError{};

  void SetUp() override
  {
#if UCXX_ENABLE_CCCL
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
#endif

    ucp_config_t* config{nullptr};
    ASSERT_EQ(ucp_config_read(nullptr, nullptr, &config), UCS_OK);
    ASSERT_EQ(ucp_config_modify(config, "RNDV_THRESH", "8192"), UCS_OK);

    ucp_params_t contextParams{};
    contextParams.field_mask = UCP_PARAM_FIELD_FEATURES;
    contextParams.features   = UCP_FEATURE_TAG;
    ASSERT_EQ(ucp_init(&contextParams, config, &_context), UCS_OK);
    ucp_config_release(config);

    ucp_worker_params_t workerParams{};
    ASSERT_EQ(ucp_worker_create(_context, &workerParams, &_clientWorker), UCS_OK);
    ASSERT_EQ(ucp_worker_create(_context, &workerParams, &_serverWorker), UCS_OK);

    _listenerState.worker = _serverWorker;
    sockaddr_in listenAddress{};
    listenAddress.sin_family      = AF_INET;
    listenAddress.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    listenAddress.sin_port        = htons(0);

    ucp_listener_params_t listenerParams{};
    listenerParams.field_mask =
      UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    listenerParams.sockaddr.addr    = reinterpret_cast<sockaddr*>(&listenAddress);
    listenerParams.sockaddr.addrlen = sizeof(listenAddress);
    listenerParams.conn_handler.cb  = listenerCallback;
    listenerParams.conn_handler.arg = &_listenerState;
    ASSERT_EQ(ucp_listener_create(_serverWorker, &listenerParams, &_listener), UCS_OK);

    ucp_listener_attr_t listenerAttr{};
    listenerAttr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
    ASSERT_EQ(ucp_listener_query(_listener, &listenerAttr), UCS_OK);

    ucp_ep_params_t endpointParams{};
    endpointParams.field_mask = UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                UCP_EP_PARAM_FIELD_ERR_HANDLER;
    endpointParams.sockaddr.addr    = reinterpret_cast<sockaddr*>(&listenerAttr.sockaddr);
    endpointParams.sockaddr.addrlen = sizeof(sockaddr_in);
    endpointParams.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    endpointParams.err_handler.cb   = endpointErrorCallback;
    endpointParams.err_handler.arg  = &_clientEndpointError;
    ASSERT_EQ(ucp_ep_create(_clientWorker, &endpointParams, &_clientEndpoint), UCS_OK);

    ASSERT_TRUE(progressUntil([this]() { return _listenerState.endpoint != nullptr; }));
    ASSERT_EQ(_listenerState.status, UCS_OK);
  }

  void TearDown() override
  {
    closeEndpoint(_clientEndpoint, _clientWorker);
    _clientEndpoint = nullptr;
    closeEndpoint(_listenerState.endpoint, _serverWorker);
    _listenerState.endpoint = nullptr;
    if (_listener != nullptr) ucp_listener_destroy(_listener);
    if (_clientWorker != nullptr) ucp_worker_destroy(_clientWorker);
    if (_serverWorker != nullptr) ucp_worker_destroy(_serverWorker);
    if (_context != nullptr) ucp_cleanup(_context);
  }

  template <typename Predicate>
  bool progressUntil(Predicate predicate,
                     std::chrono::milliseconds timeout = std::chrono::seconds(5))
  {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      ucp_worker_progress(_serverWorker);
      if (predicate()) return true;
      ucp_worker_progress(_clientWorker);
      if (predicate()) return true;
    }
    return false;
  }

  void closeEndpoint(ucp_ep_h endpoint, ucp_worker_h worker)
  {
    if (endpoint == nullptr) return;

    ucp_request_param_t closeParams{};
    closeParams.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    closeParams.flags        = UCP_EP_CLOSE_FLAG_FORCE;
    auto* request            = ucp_ep_close_nbx(endpoint, &closeParams);
    if (!UCS_PTR_IS_PTR(request)) return;

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (ucp_request_check_status(request) == UCS_INPROGRESS &&
           std::chrono::steady_clock::now() < deadline)
      ucp_worker_progress(worker);
    ucp_request_free(request);
  }

  static bool areCompleted(const std::vector<Completion>& completions)
  {
    for (const auto& completion : completions)
      if (!completion.completed) return false;
    return true;
  }

  static bool areCompleted(const ReceiveState& receiveState)
  {
    return receiveState.header.completed &&
           (receiveState.header.status != UCS_OK || receiveState.postStatus != UCS_OK ||
            areCompleted(receiveState.frames));
  }

  static void expectSuccess(const std::vector<Completion>& completions)
  {
    for (size_t i = 0; i < completions.size(); ++i) {
      ASSERT_TRUE(completions[i].completed) << "message " << i;
      EXPECT_EQ(completions[i].status, UCS_OK) << "message " << i;
    }
  }
};

#if UCXX_ENABLE_CCCL
TEST_F(RawUcxTagCloseTest, PostedTagReceivesCompleteAfterSenderForceClose)
{
  constexpr size_t MaxAttempts{100};

  for (size_t attempt = 0; attempt < MaxAttempts; ++attempt) {
    std::vector<void*> sendBuffers(NumMessages, nullptr);
    std::vector<void*> recvBuffers(NumMessages, nullptr);
    std::array<char, 64> headerRecvBuffer{};
    std::array<char, 64> headerSendBuffer{};
    std::vector<Completion> sends(NumMessages + 1);
    ReceiveState receiveState{};
    receiveState.worker  = _clientWorker;
    receiveState.buffers = recvBuffers;
    for (size_t i = 0; i < NumMessages; ++i) {
      ASSERT_EQ(cudaMalloc(&sendBuffers[i], MessageSize), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&recvBuffers[i], MessageSize), cudaSuccess);
      ASSERT_EQ(cudaMemset(sendBuffers[i], static_cast<int>(i + 1), MessageSize), cudaSuccess);
    }

    ucp_request_param_t headerRecvParams{};
    headerRecvParams.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    headerRecvParams.cb.recv      = headerRecvCallback;
    headerRecvParams.user_data    = &receiveState;
    const auto headerRecvRequest  = ucp_tag_recv_nbx(_clientWorker,
                                                    headerRecvBuffer.data(),
                                                    headerRecvBuffer.size(),
                                                    Tag,
                                                    UINT64_MAX,
                                                    &headerRecvParams);
    ASSERT_FALSE(UCS_PTR_IS_ERR(headerRecvRequest));
    if (headerRecvRequest == nullptr) {
      receiveState.header.completed = true;
      receiveState.header.status    = UCS_OK;
    }

    ucp_request_param_t sendParams{};
    sendParams.op_attr_mask      = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    sendParams.cb.send           = sendCallback;
    sendParams.user_data         = &sends[0];
    const auto headerSendRequest = ucp_tag_send_nbx(
      _listenerState.endpoint, headerSendBuffer.data(), headerSendBuffer.size(), Tag, &sendParams);
    ASSERT_FALSE(UCS_PTR_IS_ERR(headerSendRequest));
    if (headerSendRequest == nullptr) {
      sends[0].completed = true;
      sends[0].status    = UCS_OK;
    }

    for (size_t i = 0; i < NumMessages; ++i) {
      sendParams.user_data = &sends[i + 1];
      const auto request =
        ucp_tag_send_nbx(_listenerState.endpoint, sendBuffers[i], MessageSize, Tag, &sendParams);
      ASSERT_FALSE(UCS_PTR_IS_ERR(request));
      if (request == nullptr) {
        sends[i + 1].completed = true;
        sends[i + 1].status    = UCS_OK;
      }
    }

    ASSERT_TRUE(progressUntil([&sends]() { return RawUcxTagCloseTest::areCompleted(sends); }));
    expectSuccess(sends);

    if (areCompleted(receiveState)) {
      ASSERT_EQ(receiveState.header.status, UCS_OK);
      ASSERT_EQ(receiveState.postStatus, UCS_OK);
      expectSuccess(receiveState.frames);
      for (auto buffer : sendBuffers)
        ASSERT_EQ(cudaFree(buffer), cudaSuccess);
      for (auto buffer : recvBuffers)
        ASSERT_EQ(cudaFree(buffer), cudaSuccess);
      continue;
    }

    closeEndpoint(_listenerState.endpoint, _serverWorker);
    _listenerState.endpoint = nullptr;

    ASSERT_TRUE(
      progressUntil([&receiveState]() { return RawUcxTagCloseTest::areCompleted(receiveState); }));
    ASSERT_EQ(receiveState.header.status, UCS_OK);
    ASSERT_EQ(receiveState.postStatus, UCS_OK);
    expectSuccess(receiveState.frames);

    for (auto buffer : sendBuffers)
      ASSERT_EQ(cudaFree(buffer), cudaSuccess);
    for (auto buffer : recvBuffers)
      ASSERT_EQ(cudaFree(buffer), cudaSuccess);
    return;
  }

  FAIL() << "sender completion with a pending posted receive was not observed in " << MaxAttempts
         << " attempts";
}
#endif

}  // namespace
