/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <numeric>
#include <ucs/memory/memory_type.h>
#include <ucs/type/status.h>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ucxx/api.h>

#include <chrono>
#include <thread>

#include "include/utils.h"

namespace {

using ::testing::Combine;
using ::testing::ContainerEq;
using ::testing::Values;

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

enum class TransferType { Am, Tag, Stream };
typedef std::vector<std::shared_ptr<ucxx::Request>> RequestContainer;

class EndpointCancelTest : public ::testing::TestWithParam<std::tuple<TransferType, size_t, bool>> {
 protected:
  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Context> _remoteContext{nullptr};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Worker> _remoteWorker{nullptr};
  TransferType _transferType{};
  size_t _messageSize{};
  RequestContainer _requests{};
  std::vector<int> _send{}, _recv{};
  bool _rndv{false};

  virtual void SetUp()
  {
    std::tie(_transferType, _messageSize, _rndv) = GetParam();

    _send.resize(_messageSize);
    _recv.resize(_messageSize, 0);
    std::iota(_send.begin(), _send.end(), 0);

    _context       = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
    _remoteContext = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
    _worker        = _context->createWorker();
    _remoteWorker  = _remoteContext->createWorker();
  }

  RequestContainer buildPair(std::shared_ptr<ucxx::Endpoint> sendEp,
                             std::shared_ptr<ucxx::Endpoint> recvEp)
  {
    if (_transferType == TransferType::Tag) {
      return RequestContainer{
        sendEp->tagSend(_send.data(), _send.size() * sizeof(int), ucxx::Tag{0}),
        recvEp->tagRecv(_recv.data(), _recv.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull)};
    } else if (_transferType == TransferType::Am) {
      return RequestContainer{
        sendEp->amSend(_send.data(), _send.size() * sizeof(int), UCS_MEMORY_TYPE_HOST),
        recvEp->amRecv()};
    } else if (_transferType == TransferType::Stream) {
      return RequestContainer{sendEp->streamSend(_send.data(), _send.size() * sizeof(int)),
                              recvEp->streamRecv(_recv.data(), _recv.size() * sizeof(int))};
    }
    return RequestContainer{};
  }
};

static void wireup(std::shared_ptr<ucxx::Endpoint> ep1,
                   std::shared_ptr<ucxx::Endpoint> ep2,
                   std::function<void()> progressWorker)
{
  // wireup
  std::vector<int> wireupSend(1, 99);
  std::vector<int> wireupRecv(wireupSend.size(), 0);

  std::vector<std::shared_ptr<ucxx::Request>> wireupRequests;
  wireupRequests.push_back(
    ep1->tagSend(wireupSend.data(), wireupSend.size() * sizeof(int), ucxx::Tag{0}));
  wireupRequests.push_back(ep2->tagRecv(
    wireupRecv.data(), wireupRecv.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull));

  while (!wireupRequests[0]->isCompleted() || !wireupRequests[1]->isCompleted())
    progressWorker();

  ASSERT_EQ(wireupRequests[0]->getStatus(), UCS_OK);
  ASSERT_EQ(wireupRequests[1]->getStatus(), UCS_OK);
  ASSERT_THAT(wireupRecv, ContainerEq(wireupSend));
}

static size_t countIncomplete(const std::vector<std::shared_ptr<ucxx::Request>>& requests)
{
  return std::count_if(requests.begin(), requests.end(), [](auto r) { return !r->isCompleted(); });
}

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

// TEST_F(EndpointTest, CancelCloseNonBlocking)
// {
//   auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
//   _worker->progress();

//   ASSERT_TRUE(ep->isAlive());

//   std::vector<int> send(100 * 1024 * 1024, 42);
//   std::vector<int> recv(send.size(), 0);

//   // Submit and wait for transfers to complete
//   std::vector<std::shared_ptr<ucxx::Request>> requests;
//   requests.push_back(ep->tagSend(send.data(), send.size() * sizeof(int), ucxx::Tag{0}));
//   requests.push_back(
//     ep->tagRecv(recv.data(), recv.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull));
//   // waitRequests(_worker, requests, _progressWorker);

//   // copyResults();

//   // Assert data correctness
//   // ASSERT_THAT(_recv[0], ContainerEq(_send[0]));

//   size_t inflightCount = 0;
//   for (const auto& req : requests) {
//     inflightCount += !req->isCompleted();
//   }
//   std::cout << "inflightCount: " << inflightCount << std::endl;

//   size_t tmpCount = 0;
//   do {
//     tmpCount = 0;
//     for (const auto& req : requests) {
//       tmpCount += req->isCompleted();
//       req->cancel();
//     }
//     _worker->progress();
//     std::cout << "tmpCount: " << tmpCount << std::endl;

//     std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//   } while (tmpCount != 2);

//   auto canceling = ep->cancelInflightRequests();
//   // ASSERT_EQ(canceling, inflightCount);
//   std::cout << canceling << std::endl;

//   while (ep->getCancelingSize() > 0) {
//     std::cout << ep->getCancelingSize() << std::endl;
//     _worker->progress();

//     std::this_thread::sleep_for(std::chrono::milliseconds(1000));

//     for (const auto& req : requests)
//       std::cout << req->isCompleted() << " ";
//     std::cout << std::endl;
//   }

//   auto close = ep->close();
//   while (!close->isCompleted())
//     _worker->progress();
//   ASSERT_EQ(close->getStatus(), UCS_OK);
// }

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

// TEST_F(EndpointTest, CancelTmp)
// {
//   auto progressWorker       = getProgressFunction(_worker, ProgressMode::Blocking);
//   auto progressRemoteWorker = getProgressFunction(_remoteWorker, ProgressMode::Blocking);
//   auto progressAllWorkers   = [&progressWorker, progressRemoteWorker]() {
//     progressWorker();
//     progressRemoteWorker();
//   };

//   auto ep = _worker->createEndpointFromWorkerAddress(_remoteWorker->getAddress());
//   while (!ep->isAlive())
//     progressWorker();

//   auto remoteEp = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
//   while (!remoteEp->isAlive())
//     progressRemoteWorker();

//   std::vector<int> send(100 * 1024 * 1024, 42);
//   std::vector<int> recv(send.size(), 0);

//   // Submit and wait for transfers to complete
//   std::vector<std::shared_ptr<ucxx::Request>> requests, remoteRequests;
//   requests.push_back(ep->tagSend(send.data(), send.size() * sizeof(int), ucxx::Tag{0}));
//   remoteRequests.push_back(remoteEp->tagRecv(recv.data(), recv.size() * sizeof(int),
//   ucxx::Tag{0}, ucxx::TagMaskFull));
//   // requests.push_back(_remoteWorker->tagRecv(
//   //   recv.data(), recv.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull));

//   // copyResults();

//   // Assert data correctness
//   // ASSERT_THAT(_recv[0], ContainerEq(_send[0]));

//   // Endpoints that are _receiving_ may close before the request completes
//   ASSERT_FALSE(requests[0]->isCompleted());
//   ASSERT_FALSE(remoteRequests[0]->isCompleted());

//   // ep->cancelInflightRequests();
//   // remoteEp->cancelInflightRequests();

//   // ASSERT_FALSE(requests[0]->isCompleted());
//   // ASSERT_FALSE(remoteRequests[0]->isCompleted());

//   while (!requests[0]->isCompleted() || ! remoteRequests[0]->isCompleted()) {
//     std::cout << "progress: " << requests[0]->isCompleted() << " " <<
//     remoteRequests[0]->isCompleted() << std::endl; progressWorker(); progressRemoteWorker();
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//   }

//   ASSERT_TRUE(requests[0]->isCompleted());
//   ASSERT_TRUE(remoteRequests[0]->isCompleted());
//   ASSERT_EQ(requests[0]->getStatus(), UCS_OK);
//   ASSERT_EQ(remoteRequests[0]->isCompleted(), UCS_OK);

//   ASSERT_THAT(recv, ContainerEq(send));
// }

TEST_F(EndpointTest, CancelSingleTmp)
{
  // auto progressWorker       = getProgressFunction(_worker, ProgressMode::Blocking);
  auto progressWorker = getProgressFunction(_worker, ProgressMode::Polling);

  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  while (!ep->isAlive())
    progressWorker();

  wireup(ep, ep, progressWorker);

  std::vector<int> send(100 * 1024 * 1024, 42);
  std::vector<int> recv(send.size(), 0);

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(ep->tagSend(send.data(), send.size() * sizeof(int), ucxx::Tag{0}));
  requests.push_back(
    ep->tagRecv(recv.data(), recv.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull));

  ASSERT_FALSE(requests[0]->isCompleted());
  ASSERT_FALSE(requests[1]->isCompleted());

  progressWorker();
  ep->cancelInflightRequests();

  // ASSERT_FALSE(requests[0]->isCompleted());
  // ASSERT_FALSE(requests[1]->isCompleted());

  while (!requests[0]->isCompleted() || !requests[1]->isCompleted()) {
    std::cout << "progress1: " << requests[0]->isCompleted() << " " << requests[1]->isCompleted()
              << std::endl;
    std::cout << "progress2: " << requests[0]->getStatus() << " " << requests[1]->getStatus()
              << std::endl;
    progressWorker();
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  std::cout << "progress3: " << requests[0]->isCompleted() << " " << requests[1]->isCompleted()
            << std::endl;
  std::cout << "progress4: " << requests[0]->getStatus() << " " << requests[1]->getStatus()
            << std::endl;

  ASSERT_TRUE(requests[0]->isCompleted());
  ASSERT_TRUE(requests[1]->isCompleted());
  ASSERT_EQ(requests[0]->getStatus(), UCS_OK);
  ASSERT_EQ(requests[1]->getStatus(), UCS_OK);

  ASSERT_THAT(recv, ContainerEq(send));
}

TEST_F(EndpointTest, StoppingRejection)
{
  auto progressWorker = getProgressFunction(_worker, ProgressMode::Blocking);

  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  while (!ep->isAlive())
    progressWorker();

  wireup(ep, ep, progressWorker);

  ep->stop();

  ASSERT_TRUE(ep->isStopping());

  std::vector<int> tmp(10, 42);

  EXPECT_THROW(ep->amSend(tmp.data(), tmp.size() * sizeof(int), UCS_MEMORY_TYPE_HOST),
               ucxx::RejectedError);
  EXPECT_THROW(ep->tagSend(tmp.data(), tmp.size() * sizeof(int), ucxx::Tag{0}),
               ucxx::RejectedError);
  EXPECT_THROW(ep->streamRecv(tmp.data(), tmp.size() * sizeof(int)), ucxx::RejectedError);
  EXPECT_THROW(ep->streamSend(tmp.data(), tmp.size() * sizeof(int)), ucxx::RejectedError);

  // TODO: tagMultiSend, memPut, memGet
}

TEST_P(EndpointCancelTest, StoppingWaitCompletionThenCancel)
{
  if (_transferType == TransferType::Stream && _messageSize == 0)
    GTEST_SKIP() << "Stream messages of size 0 are not supported.";

  auto progressWorker = getProgressFunction(_worker, ProgressMode::Blocking);

  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  while (!ep->isAlive())
    progressWorker();

  wireup(ep, ep, progressWorker);

  // Submit and wait for transfers to complete
  auto requests = buildPair(ep, ep);

  auto checkAfterSubmit = [this, &requests, &ep]() {
    // Check requests completion statuses
    if (_rndv) {
      std::for_each(
        requests.begin(), requests.end(), [](auto r) { ASSERT_FALSE(r->isCompleted()); });
    } else {
      ASSERT_TRUE(requests[0]->isCompleted());
      ASSERT_FALSE(requests[1]->isCompleted());
    }

    // Check no requests are being canceled
    ASSERT_EQ(ep->getCancelingSize(), 0);

    // Check which requests are inflight or completed
    if (_rndv)
      ASSERT_EQ(ep->getInflightSize(), requests.size());
    else
      // Eager send request completes immediately, receive request still inflight
      ASSERT_EQ(ep->getInflightSize(), 1);
  };

  // Check request statuses before stopping endpoint
  checkAfterSubmit();

  // Stop accepting new requests, except those handled by the worker
  ep->stop();
  ASSERT_TRUE(ep->isStopping());

  // Check that requests statuses haven't changed
  checkAfterSubmit();

  // Cancel inflight requests
  ep->cancelInflightRequests();
  ASSERT_EQ(ep->getCancelingSize(), countIncomplete(requests));
  ASSERT_EQ(ep->getInflightSize(), 0);

  // Wait for canceling requests to complete
  while (countIncomplete(requests) > 0)
    progressWorker();

  // Check all requests have been canceled
  std::for_each(requests.begin(), requests.end(), [](auto r) {
    auto status = r->getStatus();
    ASSERT_TRUE(status == UCS_ERR_CANCELED || status == UCS_OK);
  });

  // Check received message, if it wasn't canceled
  if (requests[1]->getStatus() == UCS_OK) {
    // Copy AM results back into a `std::vector` which can be checked with `ASSERT_THAT`
    if (_transferType == TransferType::Am) {
      auto recvBuffer = requests[1]->getRecvBuffer();
      std::copy(reinterpret_cast<int*>(recvBuffer->data()),
                reinterpret_cast<int*>(recvBuffer->data()) + _recv.size(),
                _recv.begin());
    }

    ASSERT_THAT(_recv, ContainerEq(_send));
  }

  // Check no more tracked requests exist
  ASSERT_EQ(ep->getCancelingSize(), 0);
  ASSERT_EQ(ep->getInflightSize(), 0);

  auto close = ep->close();
  ASSERT_NE(close, nullptr);
  while (!close->isCompleted())
    progressWorker();
  ASSERT_EQ(close->getStatus(), UCS_OK);
}

INSTANTIATE_TEST_SUITE_P(Eager,
                         EndpointCancelTest,
                         Combine(Values(TransferType::Tag, TransferType::Am, TransferType::Stream),
                                 Values(0, 1, 10),
                                 Values(false)));

INSTANTIATE_TEST_SUITE_P(Rndv,
                         EndpointCancelTest,
                         Combine(Values(TransferType::Tag, TransferType::Am, TransferType::Stream),
                                 Values(10485760, 104857600),
                                 Values(true)));

}  // namespace
