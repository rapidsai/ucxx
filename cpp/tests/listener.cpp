/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <ucs/config/types.h>
#include <ucs/type/status.h>
#include <vector>

#include <gtest/gtest.h>

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

}  // namespace
