/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "include/utils.h"

namespace {

using ::testing::Combine;
using ::testing::Values;

class WorkerTest : public ::testing::Test {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::createContext({}, ucxx::Context::defaultFeatureFlags)};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};

  virtual void SetUp() { _worker = _context->createWorker(); }
};

class WorkerCapabilityTest : public ::testing::Test,
                             public ::testing::WithParamInterface<std::tuple<bool, bool>> {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::createContext({}, ucxx::Context::defaultFeatureFlags)};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  bool _enableDelayedSubmission;
  bool _enableFuture;

  virtual void SetUp()
  {
    std::tie(_enableDelayedSubmission, _enableFuture) = GetParam();

    _worker = _context->createWorker(_enableDelayedSubmission, _enableFuture);
  }
};

class WorkerProgressTest : public WorkerTest,
                           public ::testing::WithParamInterface<std::tuple<bool, ProgressMode>> {
 protected:
  std::function<void()> _progressWorker;
  bool _enableDelayedSubmission;
  ProgressMode _progressMode;

  void SetUp()
  {
    std::tie(_enableDelayedSubmission, _progressMode) = GetParam();

    _worker = _context->createWorker(_enableDelayedSubmission);

    if (_progressMode == ProgressMode::Blocking)
      _worker->initBlockingProgressMode();
    else if (_progressMode == ProgressMode::ThreadPolling)
      _worker->startProgressThread(true);
    else if (_progressMode == ProgressMode::ThreadBlocking)
      _worker->startProgressThread(false);

    _progressWorker = getProgressFunction(_worker, _progressMode);
  }
};

TEST_F(WorkerTest, HandleIsValid) { ASSERT_TRUE(_worker->getHandle() != nullptr); }

TEST_P(WorkerCapabilityTest, CheckCapability)
{
  ASSERT_EQ(_worker->isDelayedRequestSubmissionEnabled(), _enableDelayedSubmission);
  ASSERT_EQ(_worker->isFutureEnabled(), _enableFuture);
}

INSTANTIATE_TEST_SUITE_P(Capabilities,
                         WorkerCapabilityTest,
                         Combine(Values(false, true), Values(false, true)));

TEST_F(WorkerTest, TagProbe)
{
  auto progressWorker = getProgressFunction(_worker, ProgressMode::Polling);
  auto ep             = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  ASSERT_FALSE(_worker->tagProbe(ucxx::Tag{0}));

  std::vector<int> buf{123};
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(ep->tagSend(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0}));
  waitRequests(_worker, requests, progressWorker);

  loopWithTimeout(std::chrono::milliseconds(5000), [this, progressWorker]() {
    progressWorker();
    return _worker->tagProbe(ucxx::Tag{0});
  });

  ASSERT_TRUE(_worker->tagProbe(ucxx::Tag{0}));
}

TEST_F(WorkerTest, AmProbe)
{
  auto progressWorker = getProgressFunction(_worker, ProgressMode::Polling);
  auto ep             = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  ASSERT_FALSE(_worker->amProbe(ep->getHandle()));

  std::vector<int> buf{123};
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(ep->amSend(buf.data(), buf.size() * sizeof(int), UCS_MEMORY_TYPE_HOST));
  waitRequests(_worker, requests, progressWorker);

  loopWithTimeout(std::chrono::milliseconds(5000), [this, progressWorker, ep]() {
    progressWorker();
    return _worker->amProbe(ep->getHandle());
  });

  ASSERT_TRUE(_worker->amProbe(ep->getHandle()));
}

TEST_P(WorkerProgressTest, ProgressAm)
{
  if (_progressMode == ProgressMode::Wait) {
    // TODO: Is this the same reason as TagMulti?
    GTEST_SKIP() << "Wait mode not supported";
  }

  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  std::vector<int> send{123};

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(ep->amSend(send.data(), send.size() * sizeof(int), UCS_MEMORY_TYPE_HOST));
  requests.push_back(ep->amRecv());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq    = requests[1];
  auto recvBuffer = recvReq->getRecvBuffer();

  ASSERT_EQ(recvBuffer->getType(), ucxx::BufferType::Host);
  ASSERT_EQ(recvBuffer->getSize(), send.size() * sizeof(int));

  std::vector<int> recvAbstract(reinterpret_cast<int*>(recvBuffer->data()),
                                reinterpret_cast<int*>(recvBuffer->data()) + send.size());
  ASSERT_EQ(recvAbstract[0], send[0]);
}

TEST_P(WorkerProgressTest, ProgressAmReceiverCallback)
{
  if (_progressMode == ProgressMode::Wait) {
    // TODO: Is this the same reason as TagMulti?
    GTEST_SKIP() << "Wait mode not supported";
  }

  // Define AM receiver callback's owner and id for callback
  ucxx::AmReceiverCallbackInfo receiverCallbackInfo("TestApp", 0);

  // Mutex required for blocking progress mode, otherwise `receivedRequests` may be
  // accessed before `push_back()` completed.
  std::mutex mutex;

  // Define AM receiver callback and register with worker
  std::vector<std::shared_ptr<ucxx::Request>> receivedRequests;
  auto callback = ucxx::AmReceiverCallbackType(
    [this, &receivedRequests, &mutex](std::shared_ptr<ucxx::Request> req) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        receivedRequests.push_back(req);
      }
    });
  _worker->registerAmReceiverCallback(receiverCallbackInfo, callback);

  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  std::vector<int> send{123};

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(
    ep->amSend(send.data(), send.size() * sizeof(int), UCS_MEMORY_TYPE_HOST, receiverCallbackInfo));
  waitRequests(_worker, requests, _progressWorker);

  while (receivedRequests.size() < 1)
    _progressWorker();

  {
    std::lock_guard<std::mutex> lock(mutex);

    auto recvReq    = receivedRequests[0];
    auto recvBuffer = recvReq->getRecvBuffer();

    ASSERT_EQ(recvBuffer->getType(), ucxx::BufferType::Host);
    ASSERT_EQ(recvBuffer->getSize(), send.size() * sizeof(int));

    std::vector<int> recvAbstract(reinterpret_cast<int*>(recvBuffer->data()),
                                  reinterpret_cast<int*>(recvBuffer->data()) + send.size());
    ASSERT_EQ(recvAbstract[0], send[0]);
  }
}

TEST_P(WorkerProgressTest, ProgressMemoryGet)
{
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  std::vector<int> send{123};
  std::vector<int> recv(1);

  size_t messageSize = send.size() * sizeof(int);

  auto memoryHandle = _context->createMemoryHandle(messageSize, send.data());

  auto localRemoteKey      = memoryHandle->createRemoteKey();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(ep, serializedRemoteKey);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(
    ep->memGet(recv.data(), messageSize, remoteKey->getBaseAddress(), remoteKey->getHandle()));
  requests.push_back(_worker->flush());
  waitRequests(_worker, requests, _progressWorker);

  ASSERT_EQ(recv[0], send[0]);
}

TEST_P(WorkerProgressTest, ProgressMemoryPut)
{
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  std::vector<int> send{123};
  std::vector<int> recv(1);

  size_t messageSize = send.size() * sizeof(int);

  auto memoryHandle = _context->createMemoryHandle(messageSize, recv.data());

  auto localRemoteKey      = memoryHandle->createRemoteKey();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(ep, serializedRemoteKey);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(
    ep->memPut(send.data(), messageSize, remoteKey->getBaseAddress(), remoteKey->getHandle()));
  requests.push_back(_worker->flush());
  waitRequests(_worker, requests, _progressWorker);

  ASSERT_EQ(recv[0], send[0]);
}

TEST_P(WorkerProgressTest, ProgressStream)
{
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  std::vector<int> send{123};
  std::vector<int> recv(1);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(ep->streamSend(send.data(), send.size() * sizeof(int), 0));
  requests.push_back(ep->streamRecv(recv.data(), recv.size() * sizeof(int), 0));
  waitRequests(_worker, requests, _progressWorker);

  ASSERT_EQ(recv[0], send[0]);
}

TEST_P(WorkerProgressTest, ProgressTag)
{
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  std::vector<int> send{123};
  std::vector<int> recv(1);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(ep->tagSend(send.data(), send.size() * sizeof(int), ucxx::Tag{0}));
  requests.push_back(
    ep->tagRecv(recv.data(), recv.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull));
  waitRequests(_worker, requests, _progressWorker);

  ASSERT_EQ(recv[0], send[0]);
}

TEST_P(WorkerProgressTest, ProgressTagMulti)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  std::vector<int> send{123};

  const size_t numMulti = 8;

  std::vector<void*> multiBuffer(numMulti, send.data());
  std::vector<size_t> multiSize(numMulti, send.size() * sizeof(int));
  std::vector<int> multiIsCUDA(numMulti, false);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(ep->tagMultiSend(multiBuffer, multiSize, multiIsCUDA, ucxx::Tag{0}, false));
  requests.push_back(ep->tagMultiRecv(ucxx::Tag{0}, ucxx::TagMaskFull, false));
  waitRequests(_worker, requests, _progressWorker);

  for (const auto& br :
       std::dynamic_pointer_cast<ucxx::RequestTagMulti>(requests[1])->_bufferRequests) {
    // br->buffer == nullptr are headers
    if (br->buffer) {
      ASSERT_EQ(br->buffer->getType(), ucxx::BufferType::Host);
      ASSERT_EQ(br->buffer->getSize(), send.size() * sizeof(int));

      std::vector<int> recvAbstract(reinterpret_cast<int*>(br->buffer->data()),
                                    reinterpret_cast<int*>(br->buffer->data()) + send.size());
      ASSERT_EQ(recvAbstract[0], send[0]);

      const auto& recvConcretePtr = std::dynamic_pointer_cast<ucxx::HostBuffer>(br->buffer);
      ASSERT_EQ(recvConcretePtr->getType(), ucxx::BufferType::Host);
      ASSERT_EQ(recvConcretePtr->getSize(), send.size() * sizeof(int));

      std::vector<int> recvConcrete(reinterpret_cast<int*>(recvConcretePtr->data()),
                                    reinterpret_cast<int*>(recvConcretePtr->data()) + send.size());
      ASSERT_EQ(recvConcrete[0], send[0]);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ProgressModes,
                         WorkerProgressTest,
                         Combine(Values(false),
                                 Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        ProgressMode::Wait,
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking)));

INSTANTIATE_TEST_SUITE_P(
  DelayedSubmission,
  WorkerProgressTest,
  Combine(Values(true), Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking)));

}  // namespace
