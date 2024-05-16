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
#include "ucxx/exception.h"
#include "ucxx/typedefs.h"

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

class EndpointCancelTest
  : public ::testing::TestWithParam<std::tuple<ProgressMode, TransferType, size_t, bool>> {
 protected:
  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Context> _remoteContext{nullptr};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Worker> _remoteWorker{nullptr};
  ProgressMode _progressMode{};
  TransferType _transferType{};
  size_t _messageSize{};
  RequestContainer _requests{};
  std::vector<int> _send{}, _recv{};
  bool _rndv{false};

  virtual void SetUp()
  {
    std::tie(_progressMode, _transferType, _messageSize, _rndv) = GetParam();

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

TEST_F(EndpointTest, StoppingRejectRequests)
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

  {
    auto memoryHandle = _context->createMemoryHandle(tmp.size() * sizeof(int), nullptr);

    auto localRemoteKey      = memoryHandle->createRemoteKey();
    auto serializedRemoteKey = localRemoteKey->serialize();
    auto remoteKey           = ucxx::createRemoteKeyFromSerialized(ep, serializedRemoteKey);

    std::vector<std::shared_ptr<ucxx::Request>> requests;
    EXPECT_THROW(ep->memPut(tmp.data(), tmp.size() * sizeof(int), remoteKey), ucxx::RejectedError);
    EXPECT_THROW(
      ep->memPut(
        tmp.data(), tmp.size() * sizeof(int), remoteKey->getBaseAddress(), remoteKey->getHandle()),
      ucxx::RejectedError);
    EXPECT_THROW(ep->memGet(tmp.data(), tmp.size() * sizeof(int), remoteKey), ucxx::RejectedError);
    EXPECT_THROW(
      ep->memGet(
        tmp.data(), tmp.size() * sizeof(int), remoteKey->getBaseAddress(), remoteKey->getHandle()),
      ucxx::RejectedError);
  }

  {
    std::vector<void*> buffers{tmp.data()};
    std::vector<size_t> sizes{tmp.size()};
    std::vector<int> isCUDA{false};
    EXPECT_THROW(ep->tagMultiSend(buffers, sizes, isCUDA, ucxx::Tag{0}), ucxx::RejectedError);
  }
}

TEST_P(EndpointCancelTest, StoppingWaitCompletionThenCancel)
{
  if (_transferType == TransferType::Stream && _messageSize == 0)
    GTEST_SKIP() << "Stream messages of size 0 are not supported.";

  // Get appropriate progress worker function depending on selected mode
  auto progressWorker = getProgressFunction(_worker, _progressMode);
  if (_progressMode == ProgressMode::ThreadPolling)
    _worker->startProgressThread(true);
  else if (_progressMode == ProgressMode::ThreadBlocking)
    _worker->startProgressThread(false);

  // Create endpoint
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  while (!ep->isAlive())
    progressWorker();

  // Perform endpoint wireup
  wireup(ep, ep, progressWorker);

  // Submit transfer requests
  auto requests = buildPair(ep, ep);

  auto checkAfterSubmit = [this, &requests, &ep]() {
    // Check requests completion statuses
    if (_rndv) {
      std::for_each(
        requests.begin(), requests.end(), [](auto r) { ASSERT_FALSE(r->isCompleted()); });
    } else {
      ASSERT_TRUE(requests[0]->isCompleted());
      // In thread progress mode it's not possible to determine completion time
      if (_progressMode != ProgressMode::ThreadBlocking &&
          _progressMode != ProgressMode::ThreadPolling)
        ASSERT_FALSE(requests[1]->isCompleted());
    }

    // Check no requests are being canceled
    ASSERT_EQ(ep->getCancelingSize(), 0);

    // Check which requests are inflight or completed
    if (_rndv) {
      ASSERT_EQ(ep->getInflightSize(), requests.size());
    } else {
      // Eager send request completes immediately, receive request still inflight, except
      // for thread progress mode where it's not possible to determine completion time
      if (_progressMode != ProgressMode::ThreadBlocking &&
          _progressMode != ProgressMode::ThreadPolling)
        ASSERT_EQ(ep->getInflightSize(), 1);
    }
  };

  // Check request statuses before stopping endpoint
  checkAfterSubmit();

  // Stop accepting new requests, except those handled by the worker
  ep->stop();
  ASSERT_TRUE(ep->isStopping());

  // Check that requests statuses haven't changed
  checkAfterSubmit();

  bool cancelationComplete    = false;
  auto cancelInflightCallback = [&ep, &progressWorker, &cancelationComplete]() {
    // No more inflight or canceling requests, closing the endpoint is safe
    auto close = ep->close();
    ASSERT_NE(close, nullptr);
    while (!close->isCompleted())
      progressWorker();
    ASSERT_EQ(close->getStatus(), UCS_OK);

    cancelationComplete = true;
  };

  // Cancel inflight requests
  ep->cancelInflightRequests(cancelInflightCallback);
  ASSERT_EQ(ep->getCancelingSize(), countIncomplete(requests));
  ASSERT_EQ(ep->getInflightSize(), 0);

  // Wait for canceling requests to complete and `cancelInflightCallback` to run
  while (!cancelationComplete)
    progressWorker();

  // `cancelInflightCallback` executed an the endpoint should be closed now.
  ASSERT_FALSE(ep->isAlive());

  // Check all requests have been canceled or completed
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
}

INSTANTIATE_TEST_SUITE_P(Eager,
                         EndpointCancelTest,
                         Combine(Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking),
                                 Values(TransferType::Tag, TransferType::Am, TransferType::Stream),
                                 Values(0, 1, 10),
                                 Values(false)));

INSTANTIATE_TEST_SUITE_P(Rndv,
                         EndpointCancelTest,
                         Combine(Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking),
                                 Values(TransferType::Tag, TransferType::Am, TransferType::Stream),
                                 Values(10485760, 104857600),
                                 Values(true)));

}  // namespace
