/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "utils.h"

namespace {

class WorkerTest : public ::testing::Test {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::createContext({}, ucxx::Context::defaultFeatureFlags)};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};

  virtual void SetUp() { _worker = _context->createWorker(); }
};

class WorkerProgressTest : public WorkerTest,
                           public ::testing::WithParamInterface<std::pair<bool, ProgressMode>> {
 protected:
  std::function<void()> _progressWorker;

  void SetUp()
  {
    auto param                   = GetParam();
    auto enableDelayedSubmission = param.first;
    auto progressMode            = param.second;

    _worker = _context->createWorker(enableDelayedSubmission);

    if (progressMode == ProgressMode::Blocking)
      _worker->initBlockingProgressMode();
    else if (progressMode == ProgressMode::ThreadPolling)
      _worker->startProgressThread(true);
    else if (progressMode == ProgressMode::ThreadBlocking)
      _worker->startProgressThread(false);

    _progressWorker = getProgressFunction(_worker, progressMode);
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

INSTANTIATE_TEST_SUITE_P(ProgressModes,
                         WorkerProgressTest,
                         testing::Values(std::make_pair(false, ProgressMode::Polling),
                                         std::make_pair(false, ProgressMode::Blocking),
                                         std::make_pair(false, ProgressMode::Wait),
                                         std::make_pair(false, ProgressMode::ThreadPolling),
                                         std::make_pair(false, ProgressMode::ThreadBlocking)));

INSTANTIATE_TEST_SUITE_P(DelayedSubmission,
                         WorkerProgressTest,
                         testing::Values(std::make_pair(true, ProgressMode::ThreadPolling),
                                         std::make_pair(true, ProgressMode::ThreadBlocking)));

}  // namespace

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
