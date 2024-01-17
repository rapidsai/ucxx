/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ucxx/api.h>

namespace {

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

  ASSERT_TRUE(ep->getHandle() != nullptr);
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

}  // namespace
