/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "utils.h"

namespace {

struct ListenerContainer {
  ucs_status_t status{UCS_OK};
  std::shared_ptr<ucxx::Worker> worker{nullptr};
  std::shared_ptr<ucxx::Listener> listener{nullptr};
  std::shared_ptr<ucxx::Endpoint> endpoint{nullptr};
  bool transferCompleted{false};
  // bool exchange{false};
};

typedef std::shared_ptr<ListenerContainer> ListenerContainerPtr;

static void listenerCallback(ucp_conn_request_h connRequest, void* arg)
{
  ListenerContainer* listenerContainer = (ListenerContainer*)arg;
  ucp_conn_request_attr_t attr{};
  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;

  listenerContainer->status = ucp_conn_request_query(connRequest, &attr);
  if (listenerContainer->status != UCS_OK) return;

  listenerContainer->endpoint =
    listenerContainer->listener->createEndpointFromConnRequest(connRequest);

  // while (transferCompleted) ;
  // std::cout << "Listener completed" << std::endl;

  // std::vector<int> buf(1);
  // auto recv_req = listenerContainer->endpoint->tagRecv(buf.data(), buf.size() * sizeof(int), 0);
  // while (!recv_req->isCompleted()) ;
  // auto send_req = listenerContainer->endpoint->tagSend(buf.data(), buf.size() * sizeof(int), 1);
  // while (!send_req->isCompleted()) ;

  // std::vector<int> buf(1);
  // auto recv_req = listenerContainer->endpoint->tagRecv(buf.data(), buf.size() * sizeof(int), 0);
  // while (!recv_req->isCompleted()) ;
  // auto send_req = listenerContainer->endpoint->tagSend(buf.data(), buf.size() * sizeof(int), 1);
  // while (!send_req->isCompleted()) ;
  // while (!send_req->isCompleted() || !recv_req->isCompleted())
  //     std::cout << "listener incomplete" << std::endl;
  // listenerContainer->transferCompleted = true;
  // std::cout << "completed" << std::endl;
  //   listenerContainer->worker->progress();

  // try {
  //   listenerContainer->endpoint =
  //   listenerContainer->worker->createEndpointFromConnRequest(conn_request);
  // } catch (const std::bad_alloc& e)
  // {
  // } catch (const ucxx::Error& e) {
  // }
  // if (ListenerContainer->sta
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
  requests.push_back(
    listenerContainer->endpoint->tagRecv(&server_buf.front(), server_buf.size() * sizeof(int), 0));
  ::waitRequests(_worker, requests, progress);

  ASSERT_EQ(server_buf[0], client_buf[0]);

  requests.push_back(
    listenerContainer->endpoint->tagSend(&server_buf.front(), server_buf.size() * sizeof(int), 1));
  requests.push_back(ep->tagRecv(client_buf.data(), client_buf.size() * sizeof(int), 1));
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
  _worker->progress();
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
  _worker->progress();
  EXPECT_THROW(ep->raiseOnError(), ucxx::Error);
}

TEST_F(ListenerTest, CloseCallback)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = createListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->createEndpointFromHostname("127.0.0.1", listener->getPort());

  bool isClosed = false;
  ep->setCloseCallback([](void* isClosed) { *(bool*)isClosed = true; }, (void*)&isClosed);

  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  ASSERT_FALSE(isClosed);

  listenerContainer->endpoint = nullptr;
  _worker->progress();

  ASSERT_TRUE(isClosed);
}

}  // namespace
