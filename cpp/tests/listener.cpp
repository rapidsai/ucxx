/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "include/utils.h"

namespace {

constexpr size_t MaxProgressAttempts = 50;
constexpr size_t MaxFlakyAttempts    = 3;

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
  requests.push_back(ep->tagSend(client_buf.data(), client_buf.size() * sizeof(int), 0));
  requests.push_back(listenerContainer->endpoint->tagRecv(
    &server_buf.front(), server_buf.size() * sizeof(int), 0, -1));
  ::waitRequests(_worker, requests, progress);

  ASSERT_EQ(server_buf[0], client_buf[0]);

  requests.push_back(
    listenerContainer->endpoint->tagSend(&server_buf.front(), server_buf.size() * sizeof(int), 1));
  requests.push_back(ep->tagRecv(client_buf.data(), client_buf.size() * sizeof(int), 1, -1));
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
  auto send_req = ep->tagSend(buf.data(), buf.size() * sizeof(int), 0);
  while (!send_req->isCompleted())
    _worker->progress();

  listenerContainer->endpoint = nullptr;
  for (size_t attempt = 0; attempt < MaxProgressAttempts && ep->isAlive(); ++attempt)
    _worker->progress();
  ASSERT_FALSE(ep->isAlive());
}

TEST_F(ListenerTest, RaiseOnError)
{
  auto run = [this](bool lastAttempt) {
    auto listenerContainer = createListenerContainer();
    auto listener          = createListener(listenerContainer);
    _worker->progress();

    auto ep = _worker->createEndpointFromHostname("127.0.0.1", listener->getPort());
    while (listenerContainer->endpoint == nullptr)
      _worker->progress();

    listenerContainer->endpoint = nullptr;
    bool success                = false;
    for (size_t attempt = 0; attempt < MaxProgressAttempts; ++attempt) {
      try {
        _worker->progress();
        ep->raiseOnError();
      } catch (ucxx::Error) {
        success = true;
        break;
      }
    }

    if (!success && !lastAttempt) return false;

    EXPECT_THROW(ep->raiseOnError(), ucxx::Error);
    return true;
  };

  for (size_t flakyAttempt = 0; flakyAttempt < MaxFlakyAttempts; ++flakyAttempt) {
    if (run(flakyAttempt == MaxFlakyAttempts - 1)) break;
  }
}

TEST_F(ListenerTest, CloseCallback)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = createListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->createEndpointFromHostname("127.0.0.1", listener->getPort());

  bool isClosed = false;
  ep->setCloseCallback([](void* isClosed) { *reinterpret_cast<bool*>(isClosed) = true; },
                       reinterpret_cast<void*>(&isClosed));

  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  ASSERT_FALSE(isClosed);

  listenerContainer->endpoint = nullptr;
  for (size_t attempt = 0; attempt < MaxProgressAttempts && !isClosed; ++attempt)
    _worker->progress();

  ASSERT_TRUE(isClosed);
}

}  // namespace
