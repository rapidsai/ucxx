/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ucxx/api.h>

#include <chrono>
#include <thread>

#include "include/utils.h"

namespace {

using ::testing::ContainerEq;

class EndpointTest : public ::testing::Test {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::createContext({}, ucxx::Context::defaultFeatureFlags)};
  std::shared_ptr<ucxx::Context> _remoteContext{
    ucxx::createContext({}, ucxx::Context::defaultFeatureFlags)};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Worker> _remoteWorker{nullptr};

  virtual void SetUp()
  {
    _worker       = _context->createWorker();
    _remoteWorker = _remoteContext->createWorker();
  }
};

TEST_F(EndpointTest, HandleIsValid)
{
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  _worker->progress();

  ASSERT_NE(ep->getHandle(), nullptr);
}

TEST_F(EndpointTest, IsAlive)
{
  GTEST_SKIP()
    << "Connecting to worker via its UCX address doesn't seem to call endpoint error handler";
  auto ep = _worker->createEndpointFromWorkerAddress(_remoteWorker->getAddress());
  _worker->progress();
  _remoteWorker->progress();

  ASSERT_TRUE(ep->isAlive());

  std::vector<int> buf{123};
  auto send_req = ep->tagSend(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0});
  while (!send_req->isCompleted())
    _worker->progress();

  _remoteWorker  = nullptr;
  _remoteContext = nullptr;
  _worker->progress();
  ASSERT_FALSE(ep->isAlive());
}

TEST_F(EndpointTest, CancelCloseNonBlocking)
{
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  _worker->progress();

  ASSERT_TRUE(ep->isAlive());

  std::vector<int> send(100 * 1024 * 1024, 42);
  std::vector<int> recv(send.size(), 0);

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(ep->tagSend(send.data(), send.size() * sizeof(int), ucxx::Tag{0}));
  requests.push_back(
    ep->tagRecv(recv.data(), recv.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull));
  // waitRequests(_worker, requests, _progressWorker);

  // copyResults();

  // Assert data correctness
  // ASSERT_THAT(_recv[0], ContainerEq(_send[0]));

  size_t inflightCount = 0;
  for (const auto& req : requests) {
    inflightCount += !req->isCompleted();
  }
  std::cout << "inflightCount: " << inflightCount << std::endl;

  size_t tmpCount = 0;
  do {
    tmpCount = 0;
    for (const auto& req : requests) {
      tmpCount += req->isCompleted();
      req->cancel();
    }
    _worker->progress();
    std::cout << "tmpCount: " << tmpCount << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  } while (tmpCount != 2);

  auto canceling = ep->cancelInflightRequests();
  // ASSERT_EQ(canceling, inflightCount);
  std::cout << canceling << std::endl;

  while (ep->getCancelingSize() > 0) {
    std::cout << ep->getCancelingSize() << std::endl;
    _worker->progress();

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    for (const auto& req : requests)
      std::cout << req->isCompleted() << " ";
    std::cout << std::endl;
  }

  auto close = ep->close();
  while (!close->isCompleted())
    _worker->progress();
  ASSERT_EQ(close->getStatus(), UCS_OK);
}

TEST_F(EndpointTest, CloseNonBlockingCancel)
{
  auto progressWorker       = getProgressFunction(_worker, ProgressMode::Blocking);
  auto progressRemoteWorker = getProgressFunction(_remoteWorker, ProgressMode::Blocking);
  auto progressAllWorkers   = [&progressWorker, progressRemoteWorker]() {
    progressWorker();
    progressRemoteWorker();
  };

  auto ep = _worker->createEndpointFromWorkerAddress(_remoteWorker->getAddress());
  while (!ep->isAlive())
    progressWorker();

  auto remoteEp = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  while (!remoteEp->isAlive())
    progressRemoteWorker();

  std::vector<int> send(100 * 1024 * 1024, 42);
  std::vector<int> recv(send.size(), 0);

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests, remoteRequests;
  requests.push_back(ep->tagSend(send.data(), send.size() * sizeof(int), ucxx::Tag{0}));
  // requests.push_back(remoteEp->tagRecv(recv.data(), recv.size() * sizeof(int), ucxx::Tag{0},
  // ucxx::TagMaskFull));
  requests.push_back(_remoteWorker->tagRecv(
    recv.data(), recv.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull));

  // copyResults();

  // Assert data correctness
  // ASSERT_THAT(_recv[0], ContainerEq(_send[0]));

  // Endpoints that are _receiving_ may close before the request completes
  waitRequests(_remoteWorker, requests, progressAllWorkers);
  auto closeRemote = remoteEp->close();
  // waitRequests(_remoteWorker, requests, progressRemoteWorker);
  waitSingleRequest(closeRemote, progressRemoteWorker);
  ASSERT_FALSE(remoteEp->isAlive());

  // Endpoints that are _sending_ may _not_ close before the request completes
  auto close = ep->close();
  waitRequests(_worker, requests, progressWorker);
  waitSingleRequest(close, progressRemoteWorker);
  ASSERT_FALSE(ep->isAlive());

  // requests.clear();
  // requests.push_back(ep->tagRecv(recv.data(), recv.size() * sizeof(int), ucxx::Tag{0},
  // ucxx::TagMaskFull)); waitRequests(_worker, requests, progressWorker);

  ASSERT_THAT(recv, ContainerEq(send));

  // size_t inflightCount = 0;
  // for (const auto& req : requests) {
  //   inflightCount += !req->isCompleted();
  // }

  // // ASSERT_EQ(canceling, inflightCount);
  // std::cout << canceling << std::endl;

  // while (ep->getCancelingSize() > 0) {
  //   std::cout << ep->getCancelingSize() << std::endl;
  //   _worker->progress();

  //   std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  //   for (const auto& req : requests)
  //     std::cout << req->isCompleted() << " ";
  //   std::cout << std::endl;
  // }

  // ASSERT_EQ(close->getStatus(), UCS_OK);

  // auto canceling = ep->cancelInflightRequests();
}

}  // namespace
