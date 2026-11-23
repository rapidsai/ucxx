/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <ucs/config/types.h>
#include <ucs/type/status.h>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#if UCXX_ENABLE_CCCL
#include <cuda_runtime_api.h>
#endif

#include <ucxx/api.h>

#include "include/utils.h"

namespace {

struct ListenerContainer {
  ucs_status_t status{UCS_OK};
  std::shared_ptr<ucxx::Worker> worker{nullptr};
  std::shared_ptr<ucxx::Listener> listener{nullptr};
  std::shared_ptr<ucxx::Endpoint> endpoint{nullptr};
  bool transferCompleted{false};
  bool endpointErrorHandling{false};
};

typedef std::shared_ptr<ListenerContainer> ListenerContainerPtr;

static void listenerCallback(ucp_conn_request_h connRequest, void* arg)
{
  ListenerContainer* listenerContainer = reinterpret_cast<ListenerContainer*>(arg);
  ucp_conn_request_attr_t attr{};
  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;

  listenerContainer->status = ucp_conn_request_query(connRequest, &attr);
  if (listenerContainer->status != UCS_OK) return;

  listenerContainer->endpoint = listenerContainer->listener->endpointBuilder(connRequest)
                                  .endpointErrorHandling(listenerContainer->endpointErrorHandling)
                                  .build();
}

class ListenerTestBase {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build()};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  bool _endpointErrorHandling{true};

  ListenerContainerPtr createListenerContainer()
  {
    auto listenerContainer                   = std::make_shared<ListenerContainer>();
    listenerContainer->worker                = _worker;
    listenerContainer->endpointErrorHandling = _endpointErrorHandling;
    return listenerContainer;
  }
};

class ListenerTest : public ListenerTestBase,
                     public ::testing::Test,
                     public ::testing::WithParamInterface<bool> {
 protected:
  virtual void SetUp()
  {
    _endpointErrorHandling = GetParam();
    _worker                = _context->workerBuilder().build();
  }

  virtual std::shared_ptr<ucxx::Listener> buildListener(ListenerContainerPtr listenerContainer)
  {
    auto listener = _worker->listenerBuilder(0, listenerCallback, listenerContainer.get()).build();
    listenerContainer->listener = listener;
    return listener;
  }
};

class ListenerPortTest : public ListenerTestBase,
                         public ::testing::Test,
                         public ::testing::WithParamInterface<uint16_t> {
 protected:
  uint16_t _port;
  virtual void SetUp()
  {
    _port   = GetParam();
    _worker = _context->workerBuilder().build();
  }
  virtual std::shared_ptr<ucxx::Listener> buildListener(ListenerContainerPtr listenerContainer)
  {
    auto listener =
      _worker->listenerBuilder(_port, listenerCallback, listenerContainer.get()).build();
    listenerContainer->listener = listener;
    return listener;
  }
};

TEST_P(ListenerTest, HandleIsValid)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  ASSERT_TRUE(listener->getHandle() != nullptr);
}

TEST_P(ListenerTest, EndpointSendRecv)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  auto progress          = getProgressFunction(_worker, ProgressMode::Polling);

  progress();

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();
  while (listenerContainer->endpoint == nullptr)
    progress();

  std::vector<std::shared_ptr<ucxx::Request>> requests;

  std::vector<int> client_buf{123};
  std::vector<int> server_buf{0};
  requests.push_back(
    ep->tagSendBuilder(client_buf.data(), client_buf.size() * sizeof(int), ucxx::Tag{0}).build());
  requests.push_back(
    listenerContainer->endpoint
      ->tagRecvBuilder(
        &server_buf.front(), server_buf.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull)
      .build());
  ::waitRequests(_worker, requests, progress);

  ASSERT_EQ(server_buf[0], client_buf[0]);

  requests.push_back(
    listenerContainer->endpoint
      ->tagSendBuilder(&server_buf.front(), server_buf.size() * sizeof(int), ucxx::Tag{1})
      .build());
  requests.push_back(
    ep->tagRecvBuilder(
        client_buf.data(), client_buf.size() * sizeof(int), ucxx::Tag{1}, ucxx::TagMaskFull)
      .build());
  ::waitRequests(_worker, requests, progress);
  ASSERT_EQ(client_buf[0], server_buf[0]);

  std::vector<int> buf{0};
}

TEST_P(ListenerTest, IsAlive)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();
  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  ASSERT_TRUE(ep->isAlive());

  std::vector<int> buf{123};
  std::shared_ptr<ucxx::Request> send_req =
    ep->tagSendBuilder(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0}).build();
  while (!send_req->isCompleted())
    _worker->progress();

  listenerContainer->endpoint = nullptr;

  loopWithTimeout(std::chrono::milliseconds(5000), [this, ep]() {
    _worker->progress();
    return !ep->isAlive();
  });

  if (_endpointErrorHandling)
    ASSERT_FALSE(ep->isAlive());
  else
    ASSERT_TRUE(ep->isAlive());
}

TEST_P(ListenerTest, RaiseOnError)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();
  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  listenerContainer->endpoint = nullptr;

  loopWithTimeout(std::chrono::milliseconds(5000), [this, ep]() {
    try {
      _worker->progress();
      ep->raiseOnError();
    } catch (ucxx::Error) {
      return true;
    }
    return false;
  });

  if (_endpointErrorHandling) EXPECT_THROW(ep->raiseOnError(), ucxx::Error);
}

TEST_P(ListenerTest, EndpointCloseCallback)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();

  struct CallbackData {
    ucs_status_t status{UCS_INPROGRESS};
    bool closed{false};
  };

  auto callbackData = std::make_shared<CallbackData>();
  ep->setCloseCallback(
    [](ucs_status_t status, ucxx::EndpointCloseCallbackUserData callbackData) {
      auto cbData    = std::static_pointer_cast<CallbackData>(callbackData);
      cbData->status = status;
      cbData->closed = true;
    },
    callbackData);

  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  ASSERT_FALSE(callbackData->closed);
  ASSERT_EQ(callbackData->status, UCS_INPROGRESS);

  listenerContainer->endpoint = nullptr;

  loopWithTimeout(std::chrono::milliseconds(5000), [this, &callbackData]() {
    _worker->progress();
    return callbackData->closed;
  });

  ASSERT_TRUE(callbackData->closed);
  EXPECT_NE(callbackData->status, UCS_INPROGRESS);
}

TEST_P(ListenerTest, EndpointNonBlockingClose)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();

  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  auto closeRequest = ep->closeBuilder().build();

  auto f = [this, &closeRequest]() {
    _worker->progress();
    return closeRequest->isCompleted();
  };
  loopWithTimeout(std::chrono::milliseconds(5000), f);

  if (_endpointErrorHandling)
    ASSERT_FALSE(ep->isAlive());
  else
    ASSERT_TRUE(ep->isAlive());
  EXPECT_NE(closeRequest->getStatus(), UCS_INPROGRESS);
}

TEST_P(ListenerTest, EndpointNonBlockingCloseWithCallbacks)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  auto closeCallback = [](ucs_status_t status, ucxx::EndpointCloseCallbackUserData data) {
    auto dataStatus = std::static_pointer_cast<bool>(data);
    *dataStatus     = status;
  };
  auto closeCallbackEndpoint = std::make_shared<ucs_status_t>(UCS_INPROGRESS);
  auto closeCallbackRequest  = std::make_shared<ucs_status_t>(UCS_INPROGRESS);

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();
  ep->setCloseCallback(closeCallback, closeCallbackEndpoint);

  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  auto closeRequest = ep->closeBuilder()
                        .pythonFuture(false)
                        .callbackFunction(closeCallback)
                        .callbackData(closeCallbackRequest)
                        .build();

  auto f = [this, &closeRequest]() {
    _worker->progress();
    return closeRequest->isCompleted();
  };
  loopWithTimeout(std::chrono::milliseconds(5000), f);

  if (_endpointErrorHandling)
    ASSERT_FALSE(ep->isAlive());
  else
    ASSERT_TRUE(ep->isAlive());
  EXPECT_NE(closeRequest->getStatus(), UCS_INPROGRESS);
  ASSERT_NE(*closeCallbackEndpoint, UCS_INPROGRESS);
  ASSERT_NE(*closeCallbackRequest, UCS_INPROGRESS);
}

INSTANTIATE_TEST_SUITE_P(EndpointErrorHandling, ListenerTest, ::testing::Values(true));

TEST_P(ListenerPortTest, Port)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  if (GetParam() == 0)
    ASSERT_GE(listener->getPort(), 1024);
  else
    ASSERT_EQ(listener->getPort(), 12345);
}

INSTANTIATE_TEST_SUITE_P(PortAssignment, ListenerPortTest, ::testing::Values(0, 12345));

class ListenerHostTest : public ListenerTestBase, public ::testing::Test {
 protected:
  void SetUp() override { _worker = _context->workerBuilder().build(); }
};

#if UCXX_ENABLE_CCCL
class ListenerCudaCloseAfterSendTest : public ::testing::Test {
 protected:
  static constexpr size_t BufferSize{1 << 20};
  static constexpr size_t NumBuffers{3};
  static constexpr ucxx::Tag Tag{0};
  static constexpr ucxx::Tag ReceiptTag{1};

  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Worker> _clientWorker{nullptr};
  std::shared_ptr<ucxx::Worker> _serverWorker{nullptr};
  ListenerContainerPtr _listenerContainer{nullptr};
  std::shared_ptr<ucxx::Listener> _listener{nullptr};
  std::shared_ptr<ucxx::Endpoint> _client{nullptr};
  std::shared_ptr<ucxx::Endpoint> _server{nullptr};

  void SetUp() override
  {
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);

    _context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags)
                 .configMap({{"RNDV_THRESH", "8192"}})
                 .build();
    _clientWorker = _context->workerBuilder().cudaBufferType(ucxx::BufferType::CCCL).build();
    _serverWorker = _context->workerBuilder().cudaBufferType(ucxx::BufferType::CCCL).build();

    _listenerContainer                        = std::make_shared<ListenerContainer>();
    _listenerContainer->worker                = _serverWorker;
    _listenerContainer->endpointErrorHandling = true;
    _listener =
      _serverWorker->listenerBuilder(0, listenerCallback, _listenerContainer.get()).build();
    _listenerContainer->listener = _listener;

    _client = _clientWorker->endpointBuilder("127.0.0.1", _listener->getPort())
                .endpointErrorHandling(true)
                .build();
    ASSERT_TRUE(loopWithTimeout(std::chrono::milliseconds(5000), [this]() {
      _clientWorker->progress();
      _serverWorker->progress();
      return _listenerContainer->endpoint != nullptr;
    }));
    ASSERT_EQ(_listenerContainer->status, UCS_OK);
    _server = _listenerContainer->endpoint;
  }

  bool waitForRequest(const std::shared_ptr<ucxx::Request>& request)
  {
    return loopWithTimeout(std::chrono::milliseconds(5000), [this, request]() {
      _clientWorker->progress();
      _serverWorker->progress();
      return request->isCompleted();
    });
  }

  void closeServer() { _server->closeBlocking(10000000000 /* 10s */); }

  std::vector<std::shared_ptr<ucxx::Buffer>> allocateCudaBuffers()
  {
    std::vector<std::shared_ptr<ucxx::Buffer>> buffers;
    for (size_t i = 0; i < NumBuffers; ++i) {
      auto buffer = ucxx::allocateBuffer(ucxx::BufferType::CCCL, BufferSize);
      EXPECT_EQ(cudaMemset(buffer->data(), static_cast<int>(i + 1), BufferSize), cudaSuccess);
      buffers.push_back(std::move(buffer));
    }
    return buffers;
  }

  std::shared_ptr<ucxx::RequestTagMulti> postMultiReceive(
    const std::shared_ptr<ucxx::Endpoint>& endpoint)
  {
    return endpoint->tagMultiRecvBuilder(Tag, ucxx::TagMaskFull).pythonFuture(false).build();
  }

  std::shared_ptr<ucxx::RequestTagMulti> postMultiSend(
    const std::shared_ptr<ucxx::Endpoint>& endpoint,
    const std::vector<std::shared_ptr<ucxx::Buffer>>& buffers)
  {
    std::vector<const void*> pointers;
    std::vector<size_t> sizes(NumBuffers, BufferSize);
    std::vector<int> isCuda(NumBuffers, true);

    for (const auto& buffer : buffers)
      pointers.push_back(buffer->data());

    return endpoint->tagMultiSendBuilder(pointers, sizes, isCuda, Tag).pythonFuture(false).build();
  }

  std::shared_ptr<ucxx::Request> postReceiptReceive(const std::shared_ptr<ucxx::Endpoint>& endpoint,
                                                    uint8_t* receipt)
  {
    return endpoint->tagRecvBuilder(receipt, sizeof(*receipt), ReceiptTag, ucxx::TagMaskFull)
      .pythonFuture(false)
      .build();
  }

  std::shared_ptr<ucxx::Request> postReceiptSend(const std::shared_ptr<ucxx::Endpoint>& endpoint)
  {
    static const uint8_t receipt{1};
    return endpoint->tagSendBuilder(&receipt, sizeof(receipt), ReceiptTag)
      .pythonFuture(false)
      .build();
  }

  static std::vector<std::shared_ptr<ucxx::Buffer>> getReceivedBuffers(
    const std::shared_ptr<ucxx::RequestTagMulti>& request)
  {
    std::vector<std::shared_ptr<ucxx::Buffer>> buffers;
    for (const auto& child : request->_bufferRequests)
      if (child->buffer) buffers.push_back(child->buffer);
    return buffers;
  }

  bool waitForUnexpectedTag()
  {
    return loopWithTimeout(std::chrono::milliseconds(5000), [this]() {
      _serverWorker->progressOnce();
      _clientWorker->progressOnce();
      return _clientWorker->tagProbe(Tag)->isMatched();
    });
  }

  void expectChildrenCompletedSuccessfully(const std::shared_ptr<ucxx::RequestTagMulti>& request)
  {
    for (size_t i = 0; i < request->_bufferRequests.size(); ++i) {
      const auto& child = request->_bufferRequests[i]->request;
      ASSERT_NE(child, nullptr) << "child " << i;
      EXPECT_TRUE(child->isCompleted()) << "child " << i;
      EXPECT_EQ(child->getStatus(), UCS_OK) << "child " << i;
    }
  }
};

TEST_F(ListenerCudaCloseAfterSendTest, TagMultiExpectedMessageConfirmsBeforeSenderClose)
{
  auto sendBuffers = allocateCudaBuffers();
  auto receive     = postMultiReceive(_client);
  uint8_t receipt{0};
  auto receiptReceive = postReceiptReceive(_server, &receipt);
  auto send           = postMultiSend(_server, sendBuffers);

  ASSERT_TRUE(waitForRequest(send));
  ASSERT_EQ(send->getStatus(), UCS_OK);
  expectChildrenCompletedSuccessfully(send);

  ASSERT_TRUE(waitForRequest(receive));
  ASSERT_EQ(receive->getStatus(), UCS_OK);
  expectChildrenCompletedSuccessfully(receive);

  auto receiptSend = postReceiptSend(_client);
  ASSERT_TRUE(waitForRequest(receiptSend));
  ASSERT_EQ(receiptSend->getStatus(), UCS_OK);
  ASSERT_TRUE(waitForRequest(receiptReceive));
  ASSERT_EQ(receiptReceive->getStatus(), UCS_OK);
  ASSERT_EQ(receipt, 1);

  closeServer();
}

TEST_F(ListenerCudaCloseAfterSendTest, TagMultiExpectedMessageCompletesBeforeSenderClose)
{
  auto sendBuffers = allocateCudaBuffers();
  auto receive     = postMultiReceive(_client);
  auto send        = postMultiSend(_server, sendBuffers);

  ASSERT_TRUE(waitForRequest(send));
  ASSERT_EQ(send->getStatus(), UCS_OK);
  ASSERT_TRUE(waitForRequest(receive));
  ASSERT_EQ(receive->getStatus(), UCS_OK);

  closeServer();

  expectChildrenCompletedSuccessfully(send);
  expectChildrenCompletedSuccessfully(receive);
}

TEST_F(ListenerCudaCloseAfterSendTest, TagMultiUnexpectedMessageConfirmsBeforeSenderClose)
{
  // Complete the client-to-server half of the echo before submitting the response.
  auto clientBuffers = allocateCudaBuffers();
  auto serverReceive = postMultiReceive(_server);
  auto clientSend    = postMultiSend(_client, clientBuffers);
  ASSERT_TRUE(waitForRequest(clientSend));
  ASSERT_EQ(clientSend->getStatus(), UCS_OK);
  ASSERT_TRUE(waitForRequest(serverReceive));
  ASSERT_EQ(serverReceive->getStatus(), UCS_OK);

  auto serverBuffers = getReceivedBuffers(serverReceive);
  ASSERT_EQ(serverBuffers.size(), NumBuffers);
  clientSend.reset();
  serverReceive.reset();

  // Mirror the coroutine ordering where the server begins its echo before the
  // client posts recv_multi(). The non-removing probe proves the header entered
  // UCX's unexpected-message queue and leaves it there for tagMultiRecv.
  uint8_t receipt{0};
  auto receiptReceive = postReceiptReceive(_server, &receipt);
  auto serverSend     = postMultiSend(_server, serverBuffers);
  ASSERT_TRUE(waitForUnexpectedTag());
  auto clientReceive = postMultiReceive(_client);

  ASSERT_TRUE(waitForRequest(serverSend));
  ASSERT_EQ(serverSend->getStatus(), UCS_OK);
  expectChildrenCompletedSuccessfully(serverSend);

  ASSERT_TRUE(waitForRequest(clientReceive));
  ASSERT_EQ(clientReceive->getStatus(), UCS_OK);
  expectChildrenCompletedSuccessfully(clientReceive);

  auto receiptSend = postReceiptSend(_client);
  ASSERT_TRUE(waitForRequest(receiptSend));
  ASSERT_EQ(receiptSend->getStatus(), UCS_OK);
  ASSERT_TRUE(waitForRequest(receiptReceive));
  ASSERT_EQ(receiptReceive->getStatus(), UCS_OK);
  ASSERT_EQ(receipt, 1);

  closeServer();
}

TEST_F(ListenerCudaCloseAfterSendTest, TagUnexpectedMessageConfirmsBeforeSenderClose)
{
  auto sendBuffer = ucxx::allocateBuffer(ucxx::BufferType::CCCL, BufferSize);
  auto recvBuffer = ucxx::allocateBuffer(ucxx::BufferType::CCCL, BufferSize);
  ASSERT_EQ(cudaMemset(sendBuffer->data(), 1, BufferSize), cudaSuccess);

  uint8_t receipt{0};
  auto receiptReceive = postReceiptReceive(_server, &receipt);
  auto send =
    _server->tagSendBuilder(sendBuffer->data(), BufferSize, Tag).pythonFuture(false).build();
  ASSERT_TRUE(waitForUnexpectedTag());
  auto receive = _client->tagRecvBuilder(recvBuffer->data(), BufferSize, Tag, ucxx::TagMaskFull)
                   .pythonFuture(false)
                   .build();

  ASSERT_TRUE(waitForRequest(send));
  ASSERT_EQ(send->getStatus(), UCS_OK);

  ASSERT_TRUE(waitForRequest(receive));
  ASSERT_EQ(receive->getStatus(), UCS_OK);

  auto receiptSend = postReceiptSend(_client);
  ASSERT_TRUE(waitForRequest(receiptSend));
  ASSERT_EQ(receiptSend->getStatus(), UCS_OK);
  ASSERT_TRUE(waitForRequest(receiptReceive));
  ASSERT_EQ(receiptReceive->getStatus(), UCS_OK);
  ASSERT_EQ(receipt, 1);

  closeServer();
}
#endif

TEST_F(ListenerHostTest, BindToHost)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = _worker->listenerBuilder(0, listenerCallback, listenerContainer.get())
                    .host("127.0.0.1")
                    .build();
  listenerContainer->listener = listener;
  auto progress               = getProgressFunction(_worker, ProgressMode::Polling);

  progress();

  ASSERT_EQ(listener->getIp(), "127.0.0.1");
  ASSERT_GE(listener->getPort(), 1024);

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort()).build();
  while (listenerContainer->endpoint == nullptr)
    progress();

  std::vector<std::shared_ptr<ucxx::Request>> requests;

  std::vector<int> client_buf{123};
  std::vector<int> server_buf{0};
  requests.push_back(
    ep->tagSendBuilder(client_buf.data(), client_buf.size() * sizeof(int), ucxx::Tag{0}).build());
  requests.push_back(
    listenerContainer->endpoint
      ->tagRecvBuilder(
        &server_buf.front(), server_buf.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull)
      .build());
  ::waitRequests(_worker, requests, progress);

  ASSERT_EQ(server_buf[0], client_buf[0]);
}

TEST_F(ListenerHostTest, BindToInvalidHost)
{
  auto listenerContainer = createListenerContainer();
  EXPECT_THROW(std::ignore = _worker->listenerBuilder(0, listenerCallback, listenerContainer.get())
                               .host("invalid-address!")
                               .build(),
               ucxx::Error);
}

enum class ListenerCloseAction {
  Accept,
  AcceptAndClose,
  Reject,
};

struct ListenerCloseContainer {
  std::mutex mutex{};
  std::condition_variable cv{};
  std::shared_ptr<ucxx::Listener> listener{nullptr};
  std::vector<std::shared_ptr<ucxx::Endpoint>> endpoints{};
  std::vector<std::string> errors{};
  size_t callbackCount{0};
  ListenerCloseAction action{ListenerCloseAction::Accept};
  std::function<void()> onCallbackEnter{};  ///< Called on entry, without holding `mutex`
};

static void listenerCloseCallback(ucp_conn_request_h connRequest, void* arg)
{
  auto container = reinterpret_cast<ListenerCloseContainer*>(arg);
  if (container->onCallbackEnter) container->onCallbackEnter();
  {
    std::lock_guard<std::mutex> lock(container->mutex);
    ++container->callbackCount;
    if (container->listener == nullptr) {
      container->errors.push_back("callback called after the listener was released");
    } else if (container->action == ListenerCloseAction::Reject) {
      container->listener->reject(connRequest);
    } else {
      try {
        container->endpoints.push_back(
          container->listener->endpointBuilder(connRequest).endpointErrorHandling(true).build());
      } catch (const ucxx::Error& e) {
        container->errors.push_back(std::string("endpoint creation failed: ") + e.what());
      }
      if (container->action == ListenerCloseAction::AcceptAndClose) container->listener->close();
    }
  }
  container->cv.notify_all();
}

class ListenerCloseTest : public ::testing::Test,
                          public ::testing::WithParamInterface<ProgressMode> {
 protected:
  // Safety net only to prevent hangs, tests synchronize on state changes.
  static constexpr std::chrono::seconds SafetyTimeout{30};

  std::shared_ptr<ucxx::Context> _context{
    ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build()};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  ProgressMode _progressMode{ProgressMode::Polling};
  ListenerCloseContainer _container{};
  std::vector<std::shared_ptr<ucxx::Endpoint>> _clients{};

  void SetUp() override
  {
    _progressMode = GetParam();
    _worker       = _context->workerBuilder().build();
    if (_progressMode == ProgressMode::ThreadPolling)
      _worker->startProgressThread(true);
    else if (_progressMode == ProgressMode::ThreadBlocking)
      _worker->startProgressThread(false);
  }

  void TearDown() override
  {
    _clients.clear();
    {
      std::lock_guard<std::mutex> lock(_container.mutex);
      _container.endpoints.clear();
      _container.listener = nullptr;
    }
    if (isProgressThreadMode()) _worker->stopProgressThread();
  }

  bool isProgressThreadMode() const
  {
    return _progressMode == ProgressMode::ThreadPolling ||
           _progressMode == ProgressMode::ThreadBlocking;
  }

  std::shared_ptr<ucxx::Listener> buildListener()
  {
    std::lock_guard<std::mutex> lock(_container.mutex);
    _container.listener = _worker->listenerBuilder(0, listenerCloseCallback, &_container).build();
    return _container.listener;
  }

  std::shared_ptr<ucxx::Endpoint> connect(uint16_t port)
  {
    auto ep = _worker->endpointBuilder("127.0.0.1", port).endpointErrorHandling(true).build();
    _clients.push_back(ep);
    return ep;
  }

  size_t callbackCount()
  {
    std::lock_guard<std::mutex> lock(_container.mutex);
    return _container.callbackCount;
  }

  std::vector<std::string> callbackErrors()
  {
    std::lock_guard<std::mutex> lock(_container.mutex);
    return _container.errors;
  }

  size_t serverEndpointCount()
  {
    std::lock_guard<std::mutex> lock(_container.mutex);
    return _container.endpoints.size();
  }

  /**
   * Wait until the listener callback has been called `count` times.
   *
   * Progress is driven explicitly in polling mode, otherwise wait for the progress thread
   * to notify.
   */
  bool waitForCallbacks(size_t count)
  {
    if (isProgressThreadMode()) {
      std::unique_lock<std::mutex> lock(_container.mutex);
      return _container.cv.wait_for(
        lock, SafetyTimeout, [this, count]() { return _container.callbackCount >= count; });
    }
    return loopWithTimeout(SafetyTimeout, [this, count]() {
      _worker->progress();
      return callbackCount() >= count;
    });
  }

  /**
   * Wait until `client` reached a terminal state or the listener callback was called
   * more than `count` times.
   *
   * The connection attempt to a closed listener is refused and the client endpoint
   * errors. Once that happened the server will never deliver a connection request for
   * it, proving the callback was not called. If the listener was not closed the
   * callback is called instead and the wait finishes too.
   *
   * The port of the closed listener cannot be taken by another listener in the
   * meantime: the accepted connection of the server endpoint, which the tests keep
   * alive, remains bound to it, and Linux never assigns a port that is still bound
   * when binding to port 0, even with `SO_REUSEADDR`.
   */
  bool waitForRejectionOrCallback(std::shared_ptr<ucxx::Endpoint> client, size_t count)
  {
    return loopWithTimeout(SafetyTimeout, [this, client, count]() {
      if (!isProgressThreadMode()) _worker->progress();
      return !client->isAlive() || callbackCount() > count;
    });
  }
};

TEST_P(ListenerCloseTest, CloseStopsCallbacks)
{
  auto listener = buildListener();
  auto port     = listener->getPort();

  connect(port);
  ASSERT_TRUE(waitForCallbacks(1));
  ASSERT_EQ(callbackErrors(), std::vector<std::string>{});
  ASSERT_EQ(serverEndpointCount(), 1u);

  ASSERT_FALSE(listener->isClosed());
  ASSERT_NE(listener->getHandle(), nullptr);

  listener->close();
  ASSERT_TRUE(listener->isClosed());
  ASSERT_EQ(listener->getHandle(), nullptr);
  ASSERT_EQ(listener->getPort(), port);

  // Closing is idempotent.
  listener->close();
  ASSERT_TRUE(listener->isClosed());

  // Only the server endpoint keeps the listener object alive.
  {
    std::lock_guard<std::mutex> lock(_container.mutex);
    _container.listener = nullptr;
  }
  std::weak_ptr<ucxx::Listener> weakListener = listener;
  listener                                   = nullptr;
  ASSERT_FALSE(weakListener.expired());

  // The callback must not be called for connections after `close()`.
  auto client = connect(port);
  ASSERT_TRUE(waitForRejectionOrCallback(client, 1));
  ASSERT_EQ(callbackCount(), 1u);
  ASSERT_EQ(callbackErrors(), std::vector<std::string>{});
  ASSERT_FALSE(client->isAlive());
}

TEST_P(ListenerCloseTest, CloseFromCallback)
{
  _container.action = ListenerCloseAction::AcceptAndClose;
  auto listener     = buildListener();
  auto port         = listener->getPort();

  connect(port);
  ASSERT_TRUE(waitForCallbacks(1));
  ASSERT_TRUE(listener->isClosed());
  ASSERT_EQ(callbackErrors(), std::vector<std::string>{});
  ASSERT_EQ(serverEndpointCount(), 1u);

  auto client = connect(port);
  ASSERT_TRUE(waitForRejectionOrCallback(client, 1));
  ASSERT_EQ(callbackCount(), 1u);
  ASSERT_EQ(callbackErrors(), std::vector<std::string>{});
  ASSERT_FALSE(client->isAlive());
}

TEST_P(ListenerCloseTest, RejectFromCallback)
{
  _container.action = ListenerCloseAction::Reject;
  auto listener     = buildListener();

  auto client = connect(listener->getPort());
  ASSERT_TRUE(waitForCallbacks(1));
  ASSERT_TRUE(waitForRejectionOrCallback(client, 1));
  ASSERT_EQ(callbackCount(), 1u);
  ASSERT_EQ(callbackErrors(), std::vector<std::string>{});
  ASSERT_EQ(serverEndpointCount(), 0u);
  ASSERT_FALSE(client->isAlive());
}

TEST_P(ListenerCloseTest, ConcurrentClose)
{
  if (!isProgressThreadMode())
    GTEST_SKIP() << "Without a progress thread `close()` must be called from the thread "
                    "progressing the worker";

  auto listener = buildListener();

  constexpr size_t numThreads = 8;
  std::mutex mutex{};
  std::condition_variable cv{};
  bool start = false;
  std::vector<int> closedAfterReturn(numThreads, -1);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < numThreads; ++i) {
    threads.emplace_back([&, i]() {
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&start]() { return start; });
      }
      listener->close();
      // Every caller must observe the listener closed once `close()` returns.
      closedAfterReturn[i] = listener->isClosed();
    });
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    start = true;
  }
  cv.notify_all();
  for (auto& t : threads)
    t.join();

  ASSERT_EQ(closedAfterReturn, std::vector<int>(numThreads, 1));
}

/**
 * Close the listener from another thread while a listener callback is executing in a
 * `progress()` call on the thread progressing the worker, without a progress thread.
 *
 * `close()` then blocks in `ucp_listener_destroy()` until the ongoing progress call
 * returns, the listener callback must still be able to use the listener meanwhile.
 */
static void closeFromOtherThreadDuringCallback(ListenerCloseContainer* container,
                                               std::shared_ptr<ucxx::Worker> worker,
                                               std::shared_ptr<ucxx::Listener> listener,
                                               std::function<void()> connect,
                                               std::chrono::seconds safetyTimeout)
{
  std::mutex mutex{};
  std::condition_variable cv{};
  bool inCallback            = false;
  bool closing               = false;
  container->onCallbackEnter = [&]() {
    std::unique_lock<std::mutex> lock(mutex);
    inCallback = true;
    cv.notify_all();
    cv.wait(lock, [&closing]() { return closing; });
    lock.unlock();
    // Give `close()` time to block waiting for this progress call to return. This only
    // makes the interleaving that used to deadlock likely, the test passes regardless.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  };

  std::atomic<bool> stop{false};
  std::thread progressThread([&]() {
    while (!stop) {
      worker->progress();
      std::this_thread::yield();
    }
  });

  connect();
  bool callbackEntered = false;
  {
    std::unique_lock<std::mutex> lock(mutex);
    callbackEntered = cv.wait_for(lock, safetyTimeout, [&inCallback]() { return inCallback; });
  }
  if (!callbackEntered) {
    stop = true;
    progressThread.join();
    FAIL() << "The listener callback was not called";
  }

  auto closeFuture = std::async(std::launch::async, [&]() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      closing = true;
    }
    cv.notify_all();
    listener->close();
  });
  if (closeFuture.wait_for(safetyTimeout) != std::future_status::ready) {
    // The threads are deadlocked and cannot be joined.
    std::cerr << "close() did not complete, deadlock" << std::endl;
    std::abort();
  }

  stop = true;
  progressThread.join();
  container->onCallbackEnter = nullptr;

  ASSERT_TRUE(listener->isClosed());
}

TEST_P(ListenerCloseTest, CloseFromOtherThreadDuringAcceptingCallback)
{
  if (isProgressThreadMode()) GTEST_SKIP() << "Requires progressing the worker explicitly";

  auto listener = buildListener();
  closeFromOtherThreadDuringCallback(
    &_container,
    _worker,
    listener,
    [this, port = listener->getPort()]() { connect(port); },
    SafetyTimeout);

  ASSERT_EQ(callbackCount(), 1u);
  ASSERT_EQ(callbackErrors(), std::vector<std::string>{});
  ASSERT_EQ(serverEndpointCount(), 1u);
}

TEST_P(ListenerCloseTest, CloseFromOtherThreadDuringRejectingCallback)
{
  if (isProgressThreadMode()) GTEST_SKIP() << "Requires progressing the worker explicitly";

  _container.action = ListenerCloseAction::Reject;
  auto listener     = buildListener();
  closeFromOtherThreadDuringCallback(
    &_container,
    _worker,
    listener,
    [this, port = listener->getPort()]() { connect(port); },
    SafetyTimeout);

  ASSERT_EQ(callbackCount(), 1u);
  ASSERT_EQ(callbackErrors(), std::vector<std::string>{});
  ASSERT_EQ(serverEndpointCount(), 0u);
}

INSTANTIATE_TEST_SUITE_P(ProgressModes,
                         ListenerCloseTest,
                         ::testing::Values(ProgressMode::Polling,
                                           ProgressMode::ThreadPolling,
                                           ProgressMode::ThreadBlocking));

}  // namespace
