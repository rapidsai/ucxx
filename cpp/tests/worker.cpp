/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "include/utils.h"

namespace {

using ::testing::Combine;
using ::testing::Values;

enum class GenericCallbackType {
  None = 0,
  Pre,
  Post,
  PrePost,
  PostPre,
};

struct ExtraParams {
  GenericCallbackType genericCallbackType{GenericCallbackType::None};
};

class WorkerTest : public ::testing::Test {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::createContext({}, ucxx::Context::defaultFeatureFlags)};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};

  virtual void SetUp() { _worker = _context->createWorker(); }

  void consumeTagMessageHandle(std::vector<int>* recv_buf, ucp_tag_message_h handle, size_t length)
  {
    ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE,
                                 .datatype     = ucp_dt_make_contig(1)};
    auto request =
      ucp_tag_msg_recv_nbx(_worker->getHandle(), recv_buf->data(), length, handle, &param);
    EXPECT_FALSE(UCS_PTR_IS_ERR(request));
    EXPECT_FALSE(UCS_PTR_IS_PTR(request));
  }
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

class WorkerProgressTest
  : public WorkerTest,
    public ::testing::WithParamInterface<std::tuple<bool, ProgressMode, ExtraParams>> {
 protected:
  std::function<void()> _progressWorker;
  bool _enableDelayedSubmission;
  ProgressMode _progressMode;
  ExtraParams _extraParams;

  void SetUp()
  {
    std::tie(_enableDelayedSubmission, _progressMode, _extraParams) = GetParam();

    _worker = _context->createWorker(_enableDelayedSubmission);

    if (_progressMode == ProgressMode::Blocking)
      _worker->initBlockingProgressMode();
    else if (_progressMode == ProgressMode::ThreadPolling)
      _worker->startProgressThread(true);
    else if (_progressMode == ProgressMode::ThreadBlocking)
      _worker->startProgressThread(false);

    _progressWorker = getProgressFunction(_worker, _progressMode);
  }

  void TearDown()
  {
    if (_progressMode == ProgressMode::ThreadPolling ||
        _progressMode == ProgressMode::ThreadBlocking)
      _worker->stopProgressThread();
  }
};

class WorkerGenericCallbackTest : public WorkerProgressTest {};

class WorkerGenericCallbackSingleTest : public WorkerProgressTest {};

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

  auto probed = _worker->tagProbe(ucxx::Tag{0});
  ASSERT_FALSE(probed->isMatched());
  EXPECT_THROW(probed->getInfo(), std::runtime_error);
  EXPECT_THROW(probed->getHandle(), std::runtime_error);

  std::vector<int> buf{123};
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(ep->tagSend(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0}));
  waitRequests(_worker, requests, progressWorker);

  loopWithTimeout(std::chrono::milliseconds(5000), [this, progressWorker]() {
    progressWorker();
    auto probed = _worker->tagProbe(ucxx::Tag{0});
    return probed->isMatched();
  });

  auto probed2 = _worker->tagProbe(ucxx::Tag{0});
  ASSERT_TRUE(probed2->isMatched());
  ASSERT_EQ(probed2->getInfo().senderTag, ucxx::Tag{0});
  ASSERT_EQ(probed2->getInfo().length, buf.size() * sizeof(int));
  EXPECT_THROW(probed2->getHandle(), std::runtime_error);
}

TEST_F(WorkerTest, TagProbeRemoveBasicFunctionality)
{
  // Test that tagProbe with remove=false works as before
  auto probe1 = _worker->tagProbe(ucxx::Tag{0}, ucxx::TagMaskFull, false);
  EXPECT_FALSE(probe1->isMatched());
  EXPECT_THROW(probe1->getInfo(), std::runtime_error);
  EXPECT_THROW(probe1->getHandle(), std::runtime_error);

  // Test that tagProbe with remove=true returns the correct structure
  auto probe2 = _worker->tagProbe(ucxx::Tag{0}, ucxx::TagMaskFull, true);
  EXPECT_FALSE(probe2->isMatched());
  EXPECT_THROW(probe2->getHandle(), std::runtime_error);
}

TEST_F(WorkerTest, TagProbeRemoveWithMessage)
{
  auto progressWorker = getProgressFunction(_worker, ProgressMode::Polling);
  auto ep             = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  // Send a message
  std::vector<int> buf{123};
  auto send_req = ep->tagSend(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0});

  // Progress until message is sent
  while (!send_req->isCompleted()) {
    _worker->progress();
  }

  // Wait for message to be available for probing
  loopWithTimeout(std::chrono::milliseconds(5000), [this, progressWorker]() {
    progressWorker();
    auto probed = _worker->tagProbe(ucxx::Tag{0});
    return probed->isMatched();
  });

  // Test that tagProbe with remove=false returns TagProbeInfo
  auto probe1 = _worker->tagProbe(ucxx::Tag{0}, ucxx::TagMaskFull, false);
  EXPECT_TRUE(probe1->isMatched());
  EXPECT_EQ(probe1->getInfo().length, buf.size() * sizeof(int));
  // Should throw when trying to get handle since it's nullptr for remove=false
  EXPECT_THROW(probe1->getHandle(), std::runtime_error);

  // Test that tagProbe with remove=true returns TagProbeInfo with handle
  auto probe2 = _worker->tagProbe(ucxx::Tag{0}, ucxx::TagMaskFull, true);
  EXPECT_TRUE(probe2->isMatched());
  EXPECT_EQ(probe2->getInfo().length, buf.size() * sizeof(int));
  EXPECT_NO_THROW(probe2->getHandle());

  // Test receiving with the message handle
  std::vector<int> recv_buf(1);
  auto recv_req = _worker->tagRecvWithHandle(recv_buf.data(), probe2);

  // Progress until message is received
  while (!recv_req->isCompleted()) {
    _worker->progress();
  }

  EXPECT_EQ(recv_buf, buf);
}

TEST_F(WorkerTest, TagProbeUnconsumedWarning)
{
  auto progressWorker = getProgressFunction(_worker, ProgressMode::Polling);
  auto ep             = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  // Send a message
  std::vector<int> buf{123};
  auto send_req = ep->tagSend(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0});

  // Progress until message is sent
  while (!send_req->isCompleted()) {
    _worker->progress();
  }

  // Wait for message to be available for probing
  loopWithTimeout(std::chrono::milliseconds(5000), [this, progressWorker]() {
    progressWorker();
    auto probed = _worker->tagProbe(ucxx::Tag{0});
    return probed->isMatched();
  });

  ucp_tag_message_h handle{nullptr};
  size_t length{0};

  // Capture stderr to check for warning message on destruction
  testing::internal::CaptureStdout();
  {
    auto probe = _worker->tagProbe(ucxx::Tag{0}, ucxx::TagMaskFull, true);
    EXPECT_TRUE(probe->isMatched());
    EXPECT_NO_THROW(probe->getHandle());
    length = probe->getInfo().length;
    handle = probe->getHandle();
    EXPECT_NE(handle, nullptr);
    // probe goes out of scope here without consuming the handle
  }
  std::string stderr_output = testing::internal::GetCapturedStdout();

  // Check that the warning message is present in stderr
  EXPECT_NE(stderr_output.find("ucxx::TagProbeInfo::~TagProbeInfo, destroying"), std::string::npos);
  EXPECT_NE(stderr_output.find("unconsumed message handle"), std::string::npos);
  EXPECT_NE(
    stderr_output.find("ucxx::Worker::tagRecvWithHandle() must be called to consume the handle."),
    std::string::npos);

  // Consume the handle via a direct UCP operation to prevent UCX warning.
  std::vector<int> recv_buf(1);
  consumeTagMessageHandle(&recv_buf, handle, length);
  EXPECT_EQ(recv_buf, buf);
}

TEST_F(WorkerTest, TagProbeReleaseHandle)
{
  auto progressWorker = getProgressFunction(_worker, ProgressMode::Polling);
  auto ep             = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  // Send a message
  std::vector<int> buf{123};
  auto send_req = ep->tagSend(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0});

  // Progress until message is sent
  while (!send_req->isCompleted()) {
    _worker->progress();
  }

  // Wait for message to be available for probing
  loopWithTimeout(std::chrono::milliseconds(5000), [this, progressWorker]() {
    progressWorker();
    auto probed = _worker->tagProbe(ucxx::Tag{0});
    return probed->isMatched();
  });

  // Create a TagProbeInfo with a handle and release it
  auto probe = _worker->tagProbe(ucxx::Tag{0}, ucxx::TagMaskFull, true);
  EXPECT_TRUE(probe->isMatched());
  ucp_tag_message_h handle = probe->releaseHandle();
  EXPECT_NE(handle, nullptr);

  // After releasing, getHandle should throw
  EXPECT_THROW(probe->getHandle(), std::runtime_error);

  // Releasing again should throw
  EXPECT_THROW(probe->releaseHandle(), std::runtime_error);

  // Consume the handle via a direct UCP operation to prevent UCX warning.
  std::vector<int> recv_buf(1);
  consumeTagMessageHandle(&recv_buf, handle, probe->getInfo().length);
  EXPECT_EQ(recv_buf, buf);
}

TEST_F(WorkerTest, TagProbeConsumeHandle)
{
  auto progressWorker = getProgressFunction(_worker, ProgressMode::Polling);
  auto ep             = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  // Send a message
  std::vector<int> buf{123};
  auto send_req = ep->tagSend(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0});

  // Progress until message is sent
  while (!send_req->isCompleted()) {
    _worker->progress();
  }

  // Wait for message to be available for probing
  loopWithTimeout(std::chrono::milliseconds(5000), [this, progressWorker]() {
    progressWorker();
    auto probed = _worker->tagProbe(ucxx::Tag{0});
    return probed->isMatched();
  });

  // Create a TagProbeInfo with a handle and use it with tagRecvWithHandle
  // This should consume the handle and NOT trigger a warning in the destructor
  {
    auto probe = _worker->tagProbe(ucxx::Tag{0}, ucxx::TagMaskFull, true);
    EXPECT_TRUE(probe->isMatched());
    EXPECT_NO_THROW(probe->getHandle());

    // Actually use the handle via tagRecvWithHandle to consume it properly
    std::vector<int> recv_buf(1);
    auto recv_req = _worker->tagRecvWithHandle(recv_buf.data(), probe);

    // Progress until message is received
    while (!recv_req->isCompleted()) {
      _worker->progress();
    }
    // probe goes out of scope here, but handle has been consumed via tagRecvWithHandle
  }
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
    [this, &receivedRequests, &mutex](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
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

  std::vector<const void*> multiBuffer(numMulti, send.data());
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

TEST_P(WorkerGenericCallbackTest, RegisterGeneric)
{
  bool done1     = false;
  bool done2     = false;
  auto callback1 = [&done1]() { done1 = true; };
  auto callback2 = [&done2]() { done2 = true; };

  if (_extraParams.genericCallbackType == GenericCallbackType::Pre) {
    ASSERT_TRUE(_worker->registerGenericPre(callback1));
    ASSERT_TRUE(done1);
  } else if (_extraParams.genericCallbackType == GenericCallbackType::Post) {
    ASSERT_TRUE(_worker->registerGenericPre(callback1));
    ASSERT_TRUE(done1);
  } else if (_extraParams.genericCallbackType == GenericCallbackType::PrePost) {
    ASSERT_TRUE(_worker->registerGenericPre(callback1));
    ASSERT_TRUE(_worker->registerGenericPost(callback2));
    ASSERT_TRUE(done1);
    ASSERT_TRUE(done2);
  } else if (_extraParams.genericCallbackType == GenericCallbackType::PostPre) {
    ASSERT_TRUE(_worker->registerGenericPost(callback1));
    ASSERT_TRUE(_worker->registerGenericPre(callback2));
    ASSERT_TRUE(done1);
    ASSERT_TRUE(done2);
  }
}

TEST_P(WorkerGenericCallbackTest, RegisterGenericCancel)
{
  bool threadStarted   = false;
  bool terminateThread = false;
  bool done            = false;
  auto callback        = [&done] { done = true; };

  std::mutex m{};
  std::condition_variable conditionVariable{};

  std::thread thread =
    std::thread([this, &threadStarted, &terminateThread, &m, &conditionVariable]() {
      auto threadCallback = [&threadStarted, &terminateThread, &m, &conditionVariable]() {
        // Allow main thread to test for generic callback cancelation.
        threadStarted = true;
        conditionVariable.notify_one();

        {
          std::unique_lock l(m);
          // Wait until the main thread had a generic callback cancelled
          conditionVariable.wait(l, [&terminateThread] { return terminateThread; });
        }
      };

      if (_extraParams.genericCallbackType == GenericCallbackType::Pre ||
          _extraParams.genericCallbackType == GenericCallbackType::PrePost) {
        ASSERT_TRUE(_worker->registerGenericPre(threadCallback));
      } else if (_extraParams.genericCallbackType == GenericCallbackType::Post ||
                 _extraParams.genericCallbackType == GenericCallbackType::PostPre) {
        ASSERT_TRUE(_worker->registerGenericPost(threadCallback));
      }
    });

  {
    std::unique_lock l(m);
    // Wait until thread starts and blocks.
    conditionVariable.wait(l, [&threadStarted] { return threadStarted; });
  }

  // The thread should be running, therefore the callback will be canceled before running.
  // Note here `PrePost`/`PostPre` order is the opposite as from `thread`.
  if (_extraParams.genericCallbackType == GenericCallbackType::Pre ||
      _extraParams.genericCallbackType == GenericCallbackType::PostPre) {
    ASSERT_FALSE(_worker->registerGenericPre(callback, 1));
  } else if (_extraParams.genericCallbackType == GenericCallbackType::Post ||
             _extraParams.genericCallbackType == GenericCallbackType::PrePost) {
    ASSERT_FALSE(_worker->registerGenericPost(callback, 1));
  }
  ASSERT_FALSE(done);

  // Unblock thread to terminate.
  terminateThread = true;
  conditionVariable.notify_one();
  thread.join();

  // Nothing should be blocking the progress thread now, the callback should succeed.
  // Note here `PrePost`/`PostPre` order is the opposite as from `thread`.
  if (_extraParams.genericCallbackType == GenericCallbackType::Pre ||
      _extraParams.genericCallbackType == GenericCallbackType::PostPre) {
    ASSERT_TRUE(_worker->registerGenericPre(callback));
  } else if (_extraParams.genericCallbackType == GenericCallbackType::Post ||
             _extraParams.genericCallbackType == GenericCallbackType::PrePost) {
    ASSERT_TRUE(_worker->registerGenericPost(callback));
  }
  ASSERT_TRUE(done);
}

TEST_P(WorkerGenericCallbackSingleTest, RegisterGenericPreUncancelable)
{
  bool terminateThread = false;
  bool match           = false;

  std::mutex m{};
  std::condition_variable conditionVariable{};

  std::thread thread = std::thread([this, &terminateThread, &m, &conditionVariable]() {
    auto threadCallback = [&terminateThread, &m, &conditionVariable]() {
      {
        std::unique_lock l(m);
        conditionVariable.wait(l, [&terminateThread] { return terminateThread; });
      }
    };

    // This will submit the callback and attempt to cancel once every 1ms,
    // a warning is logged when multiples of 10 attempts to cancel are made.
    if (_extraParams.genericCallbackType == GenericCallbackType::Pre)
      ASSERT_TRUE(_worker->registerGenericPre(threadCallback, 1000000 /* 1ms */));
    else if (_extraParams.genericCallbackType == GenericCallbackType::Post)
      ASSERT_TRUE(_worker->registerGenericPost(threadCallback, 1000000 /* 1ms */));
  });

  loopWithTimeout(std::chrono::milliseconds(5000), [&match] {
    testing::internal::CaptureStdout();

    // We need to allow some time for stdout to be populated,
    // `GetCapturedStdout()` does not return the cumulative log.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    match = ::testing::Matches(::testing::ContainsRegex(
      "Could not cancel after .* attempts, the callback has not returned and the process may stop "
      "responding."))(::testing::internal::GetCapturedStdout());
    return match;
  });

  // Unblock thread to terminate.
  terminateThread = true;
  conditionVariable.notify_one();
  thread.join();

  ASSERT_TRUE(match);
}

INSTANTIATE_TEST_SUITE_P(ProgressModes,
                         WorkerProgressTest,
                         Combine(Values(false),
                                 Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        ProgressMode::Wait,
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking),
                                 Values(ExtraParams{})));

INSTANTIATE_TEST_SUITE_P(DelayedSubmission,
                         WorkerProgressTest,
                         Combine(Values(true),
                                 Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
                                 Values(ExtraParams{})));

INSTANTIATE_TEST_SUITE_P(
  GenericCallbacks,
  WorkerGenericCallbackTest,
  Combine(Values(false, true),
          Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
          Values(ExtraParams{.genericCallbackType = GenericCallbackType::Pre},
                 ExtraParams{.genericCallbackType = GenericCallbackType::Post},
                 ExtraParams{.genericCallbackType = GenericCallbackType::PrePost},
                 ExtraParams{.genericCallbackType = GenericCallbackType::PostPre})));

INSTANTIATE_TEST_SUITE_P(
  GenericCallbacksSingle,
  WorkerGenericCallbackSingleTest,
  Combine(Values(false, true),
          Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
          Values(ExtraParams{.genericCallbackType = GenericCallbackType::Pre},
                 ExtraParams{.genericCallbackType = GenericCallbackType::Post})));

}  // namespace
