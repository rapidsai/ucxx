/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "utils.h"

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

TEST_F(WorkerTest, TagProbe)
{
  auto progressWorker = getProgressFunction(_worker, ProgressMode::Polling);
  auto ep             = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  ASSERT_FALSE(_worker->tagProbe(0));

  std::vector<int> buf{123};
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(ep->tagSend(buf.data(), buf.size() * sizeof(int), 0));
  waitRequests(_worker, requests, progressWorker);

  ASSERT_TRUE(_worker->tagProbe(0));

  // TODO: absorb warnings
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
  requests.push_back(ep->tagSend(send.data(), send.size() * sizeof(int), 0));
  requests.push_back(ep->tagRecv(recv.data(), recv.size() * sizeof(int), 0));
  waitRequests(_worker, requests, _progressWorker);

  ASSERT_EQ(recv[0], send[0]);
}

TEST_P(WorkerProgressTest, ProgressTagMulti)
{
  if (_progressMode == ProgressMode::Wait) {
    // UCP worker progressing in wait mode can't be interrupted
    GTEST_SKIP();
  }

  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  std::vector<int> send{123};

  const size_t numMulti = 8;

  std::vector<void*> multiBuffer(numMulti, send.data());
  std::vector<size_t> multiSize(numMulti, send.size() * sizeof(int));
  std::vector<int> multiIsCUDA(numMulti, false);

  std::vector<std::shared_ptr<ucxx::RequestTagMulti>> requests;
  requests.push_back(ucxx::tagMultiSend(ep, multiBuffer, multiSize, multiIsCUDA, 0, false));
  requests.push_back(ucxx::tagMultiRecv(ep, 0, false));
  waitRequestsTagMulti(_worker, requests, _progressWorker);

  for (const auto& br : requests[1]->_bufferRequests) {
    // br->buffer == nullptr are headers
    if (br->buffer) {
      ASSERT_EQ(br->buffer->getType(), ucxx::BufferType::Host);
      ASSERT_EQ(br->buffer->getSize(), send.size() * sizeof(int));

      std::vector<int> recvAbstract((int*)br->buffer->data(),
                                    (int*)br->buffer->data() + send.size());
      ASSERT_EQ(recvAbstract[0], send[0]);

      const auto& recvConcretePtr = dynamic_cast<ucxx::HostBuffer*>(br->buffer);
      ASSERT_EQ(recvConcretePtr->getType(), ucxx::BufferType::Host);
      ASSERT_EQ(recvConcretePtr->getSize(), send.size() * sizeof(int));

      std::vector<int> recvConcrete((int*)recvConcretePtr->data(),
                                    (int*)recvConcretePtr->data() + send.size());
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
