/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
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
};

typedef std::shared_ptr<ListenerContainer> ListenerContainerPtr;

static void listenerCallback(ucp_conn_request_h connRequest, void* arg)
{
  ListenerContainer* listenerContainer = reinterpret_cast<ListenerContainer*>(arg);
  ucp_conn_request_attr_t attr{};
  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;

  listenerContainer->status = ucp_conn_request_query(connRequest, &attr);
  if (listenerContainer->status != UCS_OK) return;

  listenerContainer->endpoint =
    listenerContainer->listener->createEndpointFromConnRequest(connRequest);
}

class ListenerTest : public ::testing::Test {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::createContext({}, ucxx::Context::defaultFeatureFlags)};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};

  virtual void SetUp() { _worker = _context->createWorker(); }

  ListenerContainerPtr createListenerContainer()
  {
    auto listenerContainer    = std::make_shared<ListenerContainer>();
    listenerContainer->worker = _worker;
    return listenerContainer;
  }

  virtual std::shared_ptr<ucxx::Listener> createListener(ListenerContainerPtr listenerContainer)
  {
    auto listener = _worker->createListener(0, listenerCallback, listenerContainer.get());
    listenerContainer->listener = listener;
    return listener;
  }
};

class ListenerPortTest : public ListenerTest, public ::testing::WithParamInterface<uint16_t> {
 protected:
  virtual std::shared_ptr<ucxx::Listener> createListener(ListenerContainerPtr listenerContainer)
  {
    auto listener = _worker->createListener(GetParam(), listenerCallback, listenerContainer.get());
    listenerContainer->listener = listener;
    return listener;
  }
};

TEST_F(ListenerTest, HandleIsValid)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = createListener(listenerContainer);
  _worker->progress();

  ASSERT_TRUE(listener->getHandle() != nullptr);
}

TEST_P(ListenerPortTest, Port)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = createListener(listenerContainer);
  _worker->progress();

  if (GetParam() == 0)
    ASSERT_GE(listener->getPort(), 1024);
  else
    ASSERT_EQ(listener->getPort(), 12345);
}

INSTANTIATE_TEST_SUITE_P(PortAssignment, ListenerPortTest, ::testing::Values(0, 12345));

TEST_F(ListenerTest, EndpointSendRecv)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = createListener(listenerContainer);
  auto progress          = getProgressFunction(_worker, ProgressMode::Polling);

  progress();

  auto ep = _worker->createEndpointFromHostname("127.0.0.1", listener->getPort());
  while (listenerContainer->endpoint == nullptr)
    progress();

  std::vector<std::shared_ptr<ucxx::Request>> requests;

  std::vector<int> client_buf{123};
  std::vector<int> server_buf{0};
  requests.push_back(ep->tagSend(client_buf.data(), client_buf.size() * sizeof(int), ucxx::Tag{0}));
  requests.push_back(listenerContainer->endpoint->tagRecv(
    &server_buf.front(), server_buf.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull));
  ::waitRequests(_worker, requests, progress);

  ASSERT_EQ(server_buf[0], client_buf[0]);

  requests.push_back(listenerContainer->endpoint->tagSend(
    &server_buf.front(), server_buf.size() * sizeof(int), ucxx::Tag{1}));
  requests.push_back(ep->tagRecv(
    client_buf.data(), client_buf.size() * sizeof(int), ucxx::Tag{1}, ucxx::TagMaskFull));
  ::waitRequests(_worker, requests, progress);
  ASSERT_EQ(client_buf[0], server_buf[0]);

  std::vector<int> buf{0};
}

TEST_F(ListenerTest, IsAlive)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = createListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->createEndpointFromHostname("127.0.0.1", listener->getPort());
  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  ASSERT_TRUE(ep->isAlive());

  std::vector<int> buf{123};
  auto send_req = ep->tagSend(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0});
  while (!send_req->isCompleted())
    _worker->progress();

  listenerContainer->endpoint = nullptr;

  loopWithTimeout(std::chrono::milliseconds(5000), [this, ep]() {
    _worker->progress();
    return !ep->isAlive();
  });

  ASSERT_FALSE(ep->isAlive());
}

TEST_F(ListenerTest, RaiseOnError)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = createListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->createEndpointFromHostname("127.0.0.1", listener->getPort());
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

  EXPECT_THROW(ep->raiseOnError(), ucxx::Error);
}

TEST_F(ListenerTest, EndpointCloseCallback)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = createListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->createEndpointFromHostname("127.0.0.1", listener->getPort());

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

bool checkRequestWithTimeout(std::chrono::milliseconds timeout,
                             std::shared_ptr<ucxx::Worker> worker,
                             std::shared_ptr<ucxx::Request> closeRequest)
{
  auto startTime = std::chrono::system_clock::now();
  auto endTime   = startTime + std::chrono::milliseconds(timeout);

  while (std::chrono::system_clock::now() < endTime) {
    worker->progress();
    if (closeRequest->isCompleted()) return true;
  }
  return false;
}

TEST_F(ListenerTest, EndpointNonBlockingClose)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = createListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->createEndpointFromHostname("127.0.0.1", listener->getPort());

  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  auto closeRequest = ep->closeRequest();

  /**
   * FIXME: For some reason the code below calls `_worker->progress()` from within
   * `_worker->progress()`, which is invalid in UCX. The `checkRequestWithTimeout` below
   * which is functionally equivalent has no such problem. The lambda seems to behave in
   * unexpected way here. The issue also goes away if in `Endpoint::closeRequest()` the
   * line `if (callbackFunction) callbackFunction(status, callbackData);` is commented
   * out from the `combineCallbacksFunction` lambda, even when no callback is specified
   * to `ep->closeRequest()` above.
   */
  // auto f = [this, &closeRequest]() {
  //   _worker->progress();
  //   return closeRequest->isCompleted();
  // };
  // loopWithTimeout(std::chrono::milliseconds(5000), f);

  checkRequestWithTimeout(std::chrono::milliseconds(5000), _worker, closeRequest);

  ASSERT_FALSE(ep->isAlive());
  EXPECT_NE(closeRequest->getStatus(), UCS_INPROGRESS);
}

TEST_F(ListenerTest, EndpointNonBlockingCloseWithCallbacks)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = createListener(listenerContainer);
  _worker->progress();

  auto closeCallback = [](ucs_status_t status, ucxx::EndpointCloseCallbackUserData data) {
    auto dataStatus = std::static_pointer_cast<bool>(data);
    *dataStatus     = status;
  };
  auto closeCallbackEndpoint = std::make_shared<ucs_status_t>(UCS_INPROGRESS);
  auto closeCallbackRequest  = std::make_shared<ucs_status_t>(UCS_INPROGRESS);

  auto ep = _worker->createEndpointFromHostname("127.0.0.1", listener->getPort());
  ep->setCloseCallback(closeCallback, closeCallbackEndpoint);

  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  auto closeRequest = ep->closeRequest(false, closeCallback, closeCallbackRequest);

  /**
   * FIXME: For some reason the code below calls `_worker->progress()` from within
   * `_worker->progress()`, which is invalid in UCX. The `checkRequestWithTimeout` below
   * which is functionally equivalent has no such problem. The lambda seems to behave in
   * unexpected way here. The issue also goes away if in `Endpoint::closeRequest()` the
   * line `if (callbackFunction) callbackFunction(status, callbackData);` is commented
   * out from the `combineCallbacksFunction` lambda, even when no callback is specified
   * to `ep->closeRequest()` above.
   */
  // auto f = [this, &closeRequest]() {
  //   _worker->progress();
  //   return closeRequest->isCompleted();
  // };
  // loopWithTimeout(std::chrono::milliseconds(5000), f);

  checkRequestWithTimeout(std::chrono::milliseconds(5000), _worker, closeRequest);

  ASSERT_FALSE(ep->isAlive());
  EXPECT_NE(closeRequest->getStatus(), UCS_INPROGRESS);
  ASSERT_NE(*closeCallbackEndpoint, UCS_INPROGRESS);
  ASSERT_NE(*closeCallbackRequest, UCS_INPROGRESS);
}

}  // namespace
