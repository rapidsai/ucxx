/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <ucp/api/ucp.h>
#include <ucs/debug/log_def.h>
#include <ucs/type/status.h>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "include/utils.h"
#include "ucxx/buffer.h"
#include "ucxx/constructors.h"
#include "ucxx/utils/ucx.h"

#ifndef UCXX_TESTS_ENABLE_RMM
#define UCXX_TESTS_ENABLE_RMM 0
#endif

#if UCXX_TESTS_ENABLE_RMM
#include <rmm/device_buffer.hpp>
#endif

#if UCXX_ENABLE_CCCL
#include <cuda_runtime.h>
#endif

namespace {

using ::testing::Combine;
using ::testing::ContainerEq;
using ::testing::HasSubstr;
using ::testing::Values;

// Module-level storage for the UCX log capture handler below.
// ucs_log_push_handler takes a plain C function pointer (no captures), so
// the captured message must be accessible without a closure.
char gCapturedLogMessage[1024];
std::atomic<bool> gCapturedLogMessageSet{false};

static ucs_log_func_rc_t captureLogHandler(const char* /* file */,
                                           unsigned /* line */,
                                           const char* /* function */,
                                           ucs_log_level_t level,
                                           const ucs_log_component_config_t* /* comp_conf */,
                                           const char* message,
                                           va_list /* ap */)
{
  if (level == UCS_LOG_LEVEL_WARN && message != nullptr) {
    snprintf(gCapturedLogMessage, sizeof(gCapturedLogMessage), "%s", message);
    gCapturedLogMessageSet.store(true, std::memory_order_release);
  }
  return UCS_LOG_FUNC_RC_CONTINUE;
}

typedef std::vector<int> DataContainerType;

enum class TestBufferType {
  Host,
  RMM,
  CCCL,
};

bool isCudaBufferType(TestBufferType bufferType) { return bufferType != TestBufferType::Host; }

#if UCXX_TESTS_ENABLE_RMM
class RMMTestBuffer : public ucxx::Buffer {
 private:
  std::unique_ptr<rmm::device_buffer> _buffer;

 public:
  explicit RMMTestBuffer(size_t size)
    : ucxx::Buffer(ucxx::BufferType::Invalid, size),
      _buffer{std::make_unique<rmm::device_buffer>(size, rmm::cuda_stream_default)}
  {
  }

  void* data() override
  {
    if (!_buffer) throw std::runtime_error("Invalid object");
    return _buffer->data();
  }
};
#endif

class RequestTest
  : public ::testing::TestWithParam<std::tuple<TestBufferType, bool, bool, ProgressMode, size_t>> {
 protected:
  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _ep{nullptr};
  std::function<void()> _progressWorker;

  TestBufferType _bufferType;
  ucs_memory_type_t _memoryType;
  bool _registerCustomAmAllocator;
  bool _enableDelayedSubmission;
  ProgressMode _progressMode;
  size_t _messageLength;
  size_t _messageSize;
  size_t _rndvThresh{8192};

  size_t _numBuffers{0};
  std::vector<DataContainerType> _send;
  std::vector<DataContainerType> _recv;
  std::vector<std::unique_ptr<ucxx::Buffer>> _sendBuffer;
  std::vector<std::unique_ptr<ucxx::Buffer>> _recvBuffer;
  std::vector<const void*> _sendPtr{nullptr};
  std::vector<void*> _recvPtr{nullptr};

  void buildWorker(bool enableRequestAttributes)
  {
    auto builder = ucxx::workerBuilder(_context)
                     .delayedSubmission(_enableDelayedSubmission)
                     .requestAttributes(enableRequestAttributes);

#if UCXX_ENABLE_CCCL
    if (isCudaBufferType(_bufferType)) builder.cudaBufferType(ucxx::BufferType::CCCL);
#endif

    _worker = builder.build();

    if (_progressMode == ProgressMode::Blocking) {
      _worker->initBlockingProgressMode();
    } else if (_progressMode == ProgressMode::ThreadPolling ||
               _progressMode == ProgressMode::ThreadBlocking) {
      _worker->setProgressThreadStartCallback(::createCudaContextCallback, nullptr);

      if (_progressMode == ProgressMode::ThreadPolling) _worker->startProgressThread(true);
      if (_progressMode == ProgressMode::ThreadBlocking) _worker->startProgressThread(false);
    }

    _progressWorker = getProgressFunction(_worker, _progressMode);

    _ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  }

  void rebuildWorker(bool enableRequestAttributes)
  {
    if (_worker && _worker->isProgressThreadRunning()) _worker->stopProgressThread();
    _ep.reset();
    _worker.reset();
    buildWorker(enableRequestAttributes);
  }

  void SetUp()
  {
    std::tie(_bufferType,
             _registerCustomAmAllocator,
             _enableDelayedSubmission,
             _progressMode,
             _messageLength) = GetParam();

    if (_bufferType == TestBufferType::RMM) {
#if !UCXX_TESTS_ENABLE_RMM
      GTEST_SKIP() << "UCXX tests were not built with RMM support";
#endif
    }

    if (_bufferType == TestBufferType::CCCL) {
#if !UCXX_ENABLE_CCCL
      GTEST_SKIP() << "UCXX was not built with CCCL support";
#endif
    }

    _memoryType  = isCudaBufferType(_bufferType) ? UCS_MEMORY_TYPE_CUDA : UCS_MEMORY_TYPE_HOST;
    _messageSize = _messageLength * sizeof(int);

    _context = ucxx::createContext({{"RNDV_THRESH", std::to_string(_rndvThresh)}},
                                   ucxx::Context::defaultFeatureFlags);
    buildWorker(false);
  }

  void TearDown()
  {
    if (_progressMode == ProgressMode::ThreadPolling ||
        _progressMode == ProgressMode::ThreadBlocking)
      _worker->stopProgressThread();
  }

  void allocate(const size_t numBuffers = 1, const bool allocateRecvBuffer = true)
  {
    _numBuffers = numBuffers;

    _send.resize(_numBuffers);
    _recv.resize(_numBuffers);
    _sendBuffer.resize(_numBuffers);
    _sendPtr.resize(_numBuffers);
    if (allocateRecvBuffer) {
      _recvBuffer.resize(_numBuffers);
      _recvPtr.resize(_numBuffers);
    }

    for (size_t i = 0; i < _numBuffers; ++i) {
      _send[i].resize(_messageLength);
      _recv[i].resize(_messageLength);

      std::iota(_send[i].begin(), _send[i].end(), i);

      if (_bufferType == TestBufferType::Host) {
        _sendBuffer[i] = std::make_unique<ucxx::HostBuffer>(_messageSize);
        if (allocateRecvBuffer) _recvBuffer[i] = std::make_unique<ucxx::HostBuffer>(_messageSize);
#if UCXX_TESTS_ENABLE_RMM
      } else if (_bufferType == TestBufferType::RMM) {
        _sendBuffer[i] = std::make_unique<RMMTestBuffer>(_messageSize);
        if (allocateRecvBuffer) _recvBuffer[i] = std::make_unique<RMMTestBuffer>(_messageSize);
#endif
#if UCXX_ENABLE_CCCL
      } else if (_bufferType == TestBufferType::CCCL) {
        _sendBuffer[i] = std::make_unique<ucxx::CCCLBuffer>(_messageSize);
        if (allocateRecvBuffer) _recvBuffer[i] = std::make_unique<ucxx::CCCLBuffer>(_messageSize);
#endif
      }

      copyMemoryTypeAware(_sendBuffer[i]->data(), _send[i].data(), _messageSize, false);

      _sendPtr[i] = _sendBuffer[i]->data();
      if (allocateRecvBuffer) _recvPtr[i] = _recvBuffer[i]->data();
    }
#if UCXX_TESTS_ENABLE_RMM
    if (_bufferType == TestBufferType::RMM) { rmm::cuda_stream_default.synchronize(); }
#endif
    if (_bufferType == TestBufferType::CCCL) { cudaStreamSynchronize(nullptr); }
  }

  void copyResults()
  {
    for (size_t i = 0; i < _numBuffers; ++i)
      copyMemoryTypeAware(_recv[i].data(), _recvPtr[i], _messageSize, false);
#if UCXX_TESTS_ENABLE_RMM
    if (_bufferType == TestBufferType::RMM) { rmm::cuda_stream_default.synchronize(); }
#endif
    if (_bufferType == TestBufferType::CCCL) { cudaStreamSynchronize(nullptr); }
  }

  void copyMemoryTypeAware(void* dst, const void* src, size_t size, bool synchronize = true)
  {
    if (_memoryType == UCS_MEMORY_TYPE_HOST) {
      memcpy(dst, src, size);
#if UCXX_TESTS_ENABLE_RMM
    } else if (_memoryType == UCS_MEMORY_TYPE_CUDA && _bufferType == TestBufferType::RMM) {
      RMM_CUDA_TRY(
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, rmm::cuda_stream_default.value()));
      if (synchronize) rmm::cuda_stream_default.synchronize();
#endif
    } else if (_memoryType == UCS_MEMORY_TYPE_CUDA) {
      cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, nullptr);
      if (synchronize) cudaStreamSynchronize(nullptr);
    }
  }
};

TEST_P(RequestTest, ProgressManagedAm)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_registerCustomAmAllocator && _memoryType == UCS_MEMORY_TYPE_CUDA) {
#if UCXX_ENABLE_CCCL
    _worker->registerManagedAmAllocator(UCS_MEMORY_TYPE_CUDA, [](size_t length) {
      return std::make_shared<ucxx::CCCLBuffer>(length);
    });
#else
    GTEST_SKIP() << "CCCL support is required for CUDA receive allocations";
#endif
  }

  allocate(1, false);

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amManagedSend(_sendPtr[0], _messageSize, _memoryType));
  requests.push_back(_ep->amManagedRecv());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq = requests[1];
  _recvPtr[0]  = recvReq->getRecvBuffer()->data();

  // Messages of size `_rndvThresh` or larger are rendezvous and will use the custom
  // allocator, smaller messages are eager and will always be host-allocated.
  const auto expectedRecvBufferType = (_registerCustomAmAllocator && _messageSize >= _rndvThresh)
                                        ? ucxx::BufferType::CCCL
                                        : ucxx::BufferType::Host;
  ASSERT_THAT(recvReq->getRecvBuffer()->getType(), expectedRecvBufferType);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressManagedAmIovHost)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_memoryType != UCS_MEMORY_TYPE_HOST) {
    GTEST_SKIP() << "IOV test uses host buffers for deterministic validation";
  }

  const size_t messageLength = std::max<size_t>(4, _messageLength);
  std::vector<int> send(messageLength);
  std::iota(send.begin(), send.end(), 0);

  const size_t firstSegmentLength  = messageLength / 2;
  const size_t secondSegmentLength = messageLength - firstSegmentLength;
  std::vector<ucp_dt_iov_t> iov(2);
  iov[0].buffer = send.data();
  iov[0].length = firstSegmentLength * sizeof(int);
  iov[1].buffer = send.data() + firstSegmentLength;
  iov[1].length = secondSegmentLength * sizeof(int);

  auto amSendParams       = ucxx::AmSendParams{};
  amSendParams.datatype   = UCP_DATATYPE_IOV;
  amSendParams.memoryType = UCS_MEMORY_TYPE_HOST;

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amManagedSend(iov, amSendParams));
  requests.push_back(_ep->amManagedRecv());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq    = requests[1];
  auto recvBuffer = recvReq->getRecvBuffer();
  ASSERT_EQ(recvBuffer->getType(), ucxx::BufferType::Host);
  ASSERT_EQ(recvBuffer->getSize(), messageLength * sizeof(int));

  std::vector<int> recv(reinterpret_cast<int*>(recvBuffer->data()),
                        reinterpret_cast<int*>(recvBuffer->data()) + messageLength);
  ASSERT_THAT(recv, ContainerEq(send));
}

TEST_P(RequestTest, ProgressManagedAmIovValidation)
{
  auto amSendParams       = ucxx::AmSendParams{};
  amSendParams.datatype   = UCP_DATATYPE_IOV;
  amSendParams.memoryType = UCS_MEMORY_TYPE_HOST;

  EXPECT_THROW(std::ignore = _ep->amManagedSend(std::vector<ucp_dt_iov_t>{}, amSendParams),
               std::runtime_error);

  std::vector<ucp_dt_iov_t> iovWithNullBuffer(1);
  iovWithNullBuffer[0].buffer = nullptr;
  iovWithNullBuffer[0].length = 16;
  EXPECT_THROW(std::ignore = _ep->amManagedSend(iovWithNullBuffer, amSendParams),
               std::runtime_error);

  std::vector<int> send{1, 2, 3, 4};
  std::vector<ucp_dt_iov_t> validIov(1);
  validIov[0].buffer = send.data();
  validIov[0].length = send.size() * sizeof(send[0]);

  auto wrongDatatypeParams     = amSendParams;
  wrongDatatypeParams.datatype = ucp_dt_make_contig(1);
  EXPECT_THROW(std::ignore = _ep->amManagedSend(validIov, wrongDatatypeParams), std::runtime_error);
}

TEST_P(RequestTest, ProgressManagedAmMemoryTypePolicyStrict)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  const size_t bytes = std::max(_rndvThresh + 128, sizeof(int));
  std::vector<uint8_t> send(bytes, 42);

  auto amSendParams             = ucxx::AmSendParams{};
  amSendParams.memoryType       = UCS_MEMORY_TYPE_CUDA;
  amSendParams.memoryTypePolicy = ucxx::AmSendMemoryTypePolicy::ErrorOnUnsupported;

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amManagedSend(send.data(), send.size(), amSendParams));
  requests.push_back(_ep->amManagedRecv());

  // Wait for completion without calling checkError(), since the receive request
  // is expected to complete with UCS_ERR_UNSUPPORTED.
  while (!requests[0]->isCompleted() || !requests[1]->isCompleted())
    _progressWorker();

  // When the receiver rejects a rendezvous transfer, UCX propagates the error to
  // both sides, so the send may also complete with UCS_ERR_UNSUPPORTED.
  ASSERT_EQ(requests[1]->getStatus(), UCS_ERR_UNSUPPORTED);
}

TEST_P(RequestTest, ProgressManagedAmReceiverCallback)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_registerCustomAmAllocator && _memoryType == UCS_MEMORY_TYPE_CUDA) {
#if UCXX_ENABLE_CCCL
    _worker->registerManagedAmAllocator(UCS_MEMORY_TYPE_CUDA, [](size_t length) {
      return std::make_shared<ucxx::CCCLBuffer>(length);
    });
#else
    GTEST_SKIP() << "CCCL support is required for CUDA receive allocations";
#endif
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
  _worker->registerManagedAmReceiverCallback(receiverCallbackInfo, callback);

  allocate(1, false);

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(
    _ep->amManagedSend(_sendPtr[0], _messageSize, _memoryType, receiverCallbackInfo));
  waitRequests(_worker, requests, _progressWorker);

  while (receivedRequests.size() < 1)
    _progressWorker();

  {
    std::lock_guard<std::mutex> lock(mutex);
    _recvPtr[0] = receivedRequests[0]->getRecvBuffer()->data();

    // Messages larger than `_rndvThresh` are rendezvous and will use custom allocator,
    // smaller messages are eager and will always be host-allocated.
    const auto expectedRecvBufferType = (_registerCustomAmAllocator && _messageSize >= _rndvThresh)
                                          ? ucxx::BufferType::CCCL
                                          : ucxx::BufferType::Host;
    ASSERT_THAT(receivedRequests[0]->getRecvBuffer()->getType(), expectedRecvBufferType);
  }

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressManagedAmUserHeader)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_memoryType != UCS_MEMORY_TYPE_HOST) {
    GTEST_SKIP() << "User header test uses host buffers only";
  }

  allocate(1, false);

  const std::string sentHeader = "test-header-payload-\x00\x01\x02\xff";

  auto amSendParams = ucxx::AmSendParams{};
  amSendParams.setUserHeader(sentHeader);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amManagedSend(_sendPtr[0], _messageSize, amSendParams));
  requests.push_back(_ep->amManagedRecv());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq = requests[1];
  ASSERT_EQ(recvReq->getRecvHeader(), sentHeader);

  _recvPtr[0] = recvReq->getRecvBuffer()->data();
  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressManagedAmIovUserHeader)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_memoryType != UCS_MEMORY_TYPE_HOST) {
    GTEST_SKIP() << "IOV user header test uses host buffers only";
  }

  const size_t messageLength = std::max<size_t>(4, _messageLength);
  std::vector<int> send(messageLength);
  std::iota(send.begin(), send.end(), 0);

  const size_t firstSegmentLength  = messageLength / 2;
  const size_t secondSegmentLength = messageLength - firstSegmentLength;
  std::vector<ucp_dt_iov_t> iov(2);
  iov[0].buffer = send.data();
  iov[0].length = firstSegmentLength * sizeof(int);
  iov[1].buffer = send.data() + firstSegmentLength;
  iov[1].length = secondSegmentLength * sizeof(int);

  const std::string sentHeader = "iov-user-header-data";

  auto amSendParams       = ucxx::AmSendParams{};
  amSendParams.datatype   = UCP_DATATYPE_IOV;
  amSendParams.memoryType = UCS_MEMORY_TYPE_HOST;
  amSendParams.setUserHeader(sentHeader);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amManagedSend(iov, amSendParams));
  requests.push_back(_ep->amManagedRecv());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq    = requests[1];
  auto recvBuffer = recvReq->getRecvBuffer();
  ASSERT_EQ(recvBuffer->getType(), ucxx::BufferType::Host);
  ASSERT_EQ(recvBuffer->getSize(), messageLength * sizeof(int));
  ASSERT_EQ(recvReq->getRecvHeader(), sentHeader);

  std::vector<int> recv(reinterpret_cast<int*>(recvBuffer->data()),
                        reinterpret_cast<int*>(recvBuffer->data()) + messageLength);
  ASSERT_THAT(recv, ContainerEq(send));
}

TEST_P(RequestTest, ProgressManagedAmEmptyUserHeader)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_memoryType != UCS_MEMORY_TYPE_HOST) {
    GTEST_SKIP() << "User header test uses host buffers only";
  }

  allocate(1, false);

  // Send without user header (default empty)
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amManagedSend(_sendPtr[0], _messageSize, _memoryType));
  requests.push_back(_ep->amManagedRecv());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq = requests[1];
  ASSERT_EQ(recvReq->getRecvHeader(), std::string{});

  _recvPtr[0] = recvReq->getRecvBuffer()->data();
  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressStream)
{
  allocate();

  // Submit and wait for transfers to complete
  if (_messageSize == 0) {
    EXPECT_THROW(std::ignore = _ep->streamSend(_sendPtr[0], _messageSize, 0), std::runtime_error);
    EXPECT_THROW(std::ignore = _ep->streamRecv(_recvPtr[0], _messageSize, 0), std::runtime_error);
  } else {
    std::vector<std::shared_ptr<ucxx::Request>> requests;
    requests.push_back(_ep->streamSend(_sendPtr[0], _messageSize, 0));
    requests.push_back(_ep->streamRecv(_recvPtr[0], _messageSize, 0));
    waitRequests(_worker, requests, _progressWorker);

    copyResults();

    // Assert data correctness
    ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
  }
}

TEST_P(RequestTest, ProgressTag)
{
  allocate();

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->tagSend(_sendPtr[0], _messageSize, ucxx::Tag{0}));
  requests.push_back(_ep->tagRecv(_recvPtr[0], _messageSize, ucxx::Tag{0}, ucxx::TagMaskFull));
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressTagRequestAttributes)
{
  if (_messageSize < _rndvThresh)
    GTEST_SKIP() << "Eager messages do not create a ucp_request and thus no debug info";

  rebuildWorker(true);

  allocate();

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->tagSend(_sendPtr[0], _messageSize, ucxx::Tag{0}));
  requests.push_back(_ep->tagRecv(_recvPtr[0], _messageSize, ucxx::Tag{0}, ucxx::TagMaskFull));
  waitRequests(_worker, requests, _progressWorker);

  for (const auto& request : requests) {
    auto debugString = request->queryAttributes().debugString;
    ASSERT_THAT(debugString, ::testing::HasSubstr("length " + std::to_string(_messageSize)));
    ASSERT_THAT(debugString,
                ::testing::HasSubstr(_memoryType == UCS_MEMORY_TYPE_HOST ? "host" : "cuda"));
  }

  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

class RequestAttributesDisabledTest : public ::testing::Test {
 protected:
  static constexpr size_t kMessageLength = 1024;
  static constexpr size_t kMessageSize   = kMessageLength * sizeof(int);

  std::shared_ptr<ucxx::Context> _context;
  std::shared_ptr<ucxx::Worker> _worker;
  std::shared_ptr<ucxx::Endpoint> _ep;
  std::function<void()> _progressWorker;
  std::vector<int> _sendBuf;
  std::vector<int> _recvBuf;

  void SetUp() override
  {
    _context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
    _worker  = ucxx::workerBuilder(_context).build();
    ASSERT_FALSE(_worker->isRequestAttributesEnabled());

    _ep             = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
    _progressWorker = getProgressFunction(_worker, ProgressMode::Polling);

    _sendBuf.resize(kMessageLength);
    _recvBuf.resize(kMessageLength);
    std::iota(_sendBuf.begin(), _sendBuf.end(), 0);
  }

  void expectAllThrow(const std::vector<std::shared_ptr<ucxx::Request>>& requests) const
  {
    for (const auto& request : requests) {
      EXPECT_THROW(std::ignore = request->queryAttributes(), ucxx::UnsupportedError);
    }
  }
};

TEST_F(RequestAttributesDisabledTest, Tag)
{
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->tagSend(_sendBuf.data(), kMessageSize, ucxx::Tag{0}));
  requests.push_back(_ep->tagRecv(_recvBuf.data(), kMessageSize, ucxx::Tag{0}, ucxx::TagMaskFull));
  waitRequests(_worker, requests, _progressWorker);

  expectAllThrow(requests);
  ASSERT_THAT(_recvBuf, ::testing::ContainerEq(_sendBuf));
}

TEST_F(RequestAttributesDisabledTest, Stream)
{
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->streamSend(_sendBuf.data(), kMessageSize, 0));
  requests.push_back(_ep->streamRecv(_recvBuf.data(), kMessageSize, 0));
  waitRequests(_worker, requests, _progressWorker);

  expectAllThrow(requests);
  ASSERT_THAT(_recvBuf, ::testing::ContainerEq(_sendBuf));
}

TEST_F(RequestAttributesDisabledTest, ManagedAm)
{
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amManagedSend(_sendBuf.data(), kMessageSize, UCS_MEMORY_TYPE_HOST));
  requests.push_back(_ep->amManagedRecv());
  waitRequests(_worker, requests, _progressWorker);

  expectAllThrow(requests);

  auto recvBuffer = requests[1]->getRecvBuffer();
  ASSERT_EQ(recvBuffer->getSize(), kMessageSize);
  std::vector<int> received(reinterpret_cast<int*>(recvBuffer->data()),
                            reinterpret_cast<int*>(recvBuffer->data()) + kMessageLength);
  ASSERT_THAT(received, ::testing::ContainerEq(_sendBuf));
}

TEST_F(RequestAttributesDisabledTest, MemoryGet)
{
  auto memoryHandle = _context->createMemoryHandle(kMessageSize, nullptr, UCS_MEMORY_TYPE_HOST);
  std::memcpy(
    reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _sendBuf.data(), kMessageSize);

  auto serializedRemoteKey = memoryHandle->createRemoteKey()->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey);

  std::shared_ptr<ucxx::Request> request = _ep->memGet(_recvBuf.data(), kMessageSize, remoteKey);
  std::vector<std::shared_ptr<ucxx::Request>> requests{request, _ep->flush()};
  waitRequests(_worker, requests, _progressWorker);

  expectAllThrow({request});
  ASSERT_THAT(_recvBuf, ::testing::ContainerEq(_sendBuf));
}

TEST_F(RequestAttributesDisabledTest, MemoryPut)
{
  auto memoryHandle = _context->createMemoryHandle(kMessageSize, nullptr, UCS_MEMORY_TYPE_HOST);

  auto serializedRemoteKey = memoryHandle->createRemoteKey()->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey);

  std::shared_ptr<ucxx::Request> request = _ep->memPut(_sendBuf.data(), kMessageSize, remoteKey);
  std::vector<std::shared_ptr<ucxx::Request>> requests{request, _ep->flush()};
  waitRequests(_worker, requests, _progressWorker);

  expectAllThrow({request});

  std::memcpy(
    _recvBuf.data(), reinterpret_cast<void*>(memoryHandle->getBaseAddress()), kMessageSize);
  ASSERT_THAT(_recvBuf, ::testing::ContainerEq(_sendBuf));
}

TEST_P(RequestTest, ProgressStreamRequestAttributes)
{
  if (_messageSize == 0) GTEST_SKIP() << "Stream rejects zero-length transfers";

  rebuildWorker(true);

  allocate();

  std::shared_ptr<ucxx::Request> sendRequest = _ep->streamSend(_sendPtr[0], _messageSize, 0);
  std::shared_ptr<ucxx::Request> recvRequest = _ep->streamRecv(_recvPtr[0], _messageSize, 0);
  std::vector<std::shared_ptr<ucxx::Request>> requests{sendRequest, recvRequest};
  waitRequests(_worker, requests, _progressWorker);

  try {
    auto sendDebug = sendRequest->queryAttributes().debugString;
    EXPECT_FALSE(sendDebug.empty());
    EXPECT_THAT(sendDebug, ::testing::HasSubstr("length " + std::to_string(_messageSize)));
  } catch (const ucxx::NoElemError&) {
    // Send completed inline; no UCP request handle to query.
  }

  try {
    auto recvDebug = recvRequest->queryAttributes().debugString;
    EXPECT_THAT(recvDebug, ::testing::HasSubstr("no debug info"));
  } catch (const ucxx::NoElemError&) {
    // Recv completed inline; no UCP request handle to query.
  }

  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressManagedAmRequestAttributes)
{
  if (_messageSize < _rndvThresh)
    GTEST_SKIP() << "Eager messages complete inline without a UCP request to query";
  if (_progressMode == ProgressMode::Wait)
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";

  rebuildWorker(true);

  allocate(1, false);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amManagedSend(_sendPtr[0], _messageSize, _memoryType));
  requests.push_back(_ep->amManagedRecv());
  waitRequests(_worker, requests, _progressWorker);

  for (const auto& request : requests) {
    auto debugString = request->queryAttributes().debugString;
    ASSERT_THAT(debugString, ::testing::HasSubstr("length " + std::to_string(_messageSize)));
  }

  auto recvReq = requests[1];
  _recvPtr[0]  = recvReq->getRecvBuffer()->data();
  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryGetRequestAttributes)
{
  if (_messageSize == 0) GTEST_SKIP() << "Zero-length memGet completes without a UCP request";

  rebuildWorker(true);

  allocate();

  auto memoryHandle = _context->createMemoryHandle(_messageSize, nullptr, _memoryType);
  copyMemoryTypeAware(
    reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _sendPtr[0], _messageSize);

  auto localRemoteKey      = memoryHandle->createRemoteKey();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey);

  std::shared_ptr<ucxx::Request> request = _ep->memGet(_recvPtr[0], _messageSize, remoteKey);
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(request);
  requests.push_back(_ep->flush());
  waitRequests(_worker, requests, _progressWorker);

  auto debugString = request->queryAttributes().debugString;
  ASSERT_THAT(debugString, ::testing::HasSubstr("length " + std::to_string(_messageSize)));

  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryPutRequestAttributes)
{
  if (_messageSize == 0) GTEST_SKIP() << "Zero-length memPut completes without a UCP request";

  rebuildWorker(true);

  allocate();

  auto memoryHandle = _context->createMemoryHandle(_messageSize, nullptr, _memoryType);

  auto localRemoteKey      = memoryHandle->createRemoteKey();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey);

  std::shared_ptr<ucxx::Request> request = _ep->memPut(_sendPtr[0], _messageSize, remoteKey);
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(request);
  requests.push_back(_ep->flush());
  waitRequests(_worker, requests, _progressWorker);

  try {
    auto debugString = request->queryAttributes().debugString;
    EXPECT_FALSE(debugString.empty());
    EXPECT_THAT(debugString, ::testing::HasSubstr("length " + std::to_string(_messageSize)));
  } catch (const ucxx::NoElemError&) {
    // Request completed inline; no UCP request handle to query.
  }

  copyMemoryTypeAware(
    _recvPtr[0], reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _messageSize);

  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressTagMulti)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  const size_t numMulti         = 8;
  const bool allocateRecvBuffer = false;

  if (isCudaBufferType(_bufferType) && _worker->getCudaBufferType() == ucxx::BufferType::Invalid) {
    GTEST_SKIP() << "CUDA buffer allocation support not enabled";
  }

  allocate(numMulti, allocateRecvBuffer);

  // Allocate buffers for request sizes/types
  std::vector<size_t> multiSize(numMulti, _messageSize);
  std::vector<int> multiIsCUDA(numMulti, isCudaBufferType(_bufferType));

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->tagMultiSend(_sendPtr, multiSize, multiIsCUDA, ucxx::Tag{0}, false));
  requests.push_back(_ep->tagMultiRecv(ucxx::Tag{0}, ucxx::TagMaskFull, false));
  waitRequests(_worker, requests, _progressWorker);

  auto recvRequest = requests[1];

  _recvPtr.resize(_numBuffers);
  size_t transferIdx = 0;

  // Populate recv pointers
  for (const auto& br :
       std::dynamic_pointer_cast<ucxx::RequestTagMulti>(requests[1])->_bufferRequests) {
    // br->buffer == nullptr are headers
    if (br->buffer) {
      auto expectedBufferType =
        isCudaBufferType(_bufferType) ? _worker->getCudaBufferType() : ucxx::BufferType::Host;
      ASSERT_EQ(br->buffer->getType(), expectedBufferType);
      ASSERT_EQ(br->buffer->getSize(), _messageSize);

      _recvPtr[transferIdx] = br->buffer->data();

      ++transferIdx;
    }
  }

  copyResults();

  // Assert data correctness
  for (size_t i = 0; i < numMulti; ++i)
    ASSERT_THAT(_recv[i], ContainerEq(_send[i]));
}

TEST_P(RequestTest, TagUserCallback)
{
  allocate();

  std::vector<std::shared_ptr<ucxx::Request>> requests(2);
  std::vector<ucs_status_t> requestStatus(2, UCS_INPROGRESS);

  auto checkStatus = [&requests, &requestStatus](ucs_status_t status,
                                                 ::ucxx::RequestCallbackUserData data) {
    auto idx           = *std::static_pointer_cast<size_t>(data);
    requestStatus[idx] = status;
  };

  auto sendIndex = std::make_shared<size_t>(0u);
  auto recvIndex = std::make_shared<size_t>(1u);

  // Submit and wait for transfers to complete
  requests[0] =
    _ep->tagSend(_sendPtr[0], _messageSize, ucxx::Tag{0}, false, checkStatus, sendIndex);
  requests[1] = _ep->tagRecv(
    _recvPtr[0], _messageSize, ucxx::Tag{0}, ucxx::TagMaskFull, false, checkStatus, recvIndex);
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  for (const auto request : requests)
    ASSERT_THAT(request->getStatus(), UCS_OK);
  for (const auto status : requestStatus)
    ASSERT_THAT(status, UCS_OK);

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, TagUserCallbackDiscardReturn)
{
  allocate();

  std::vector<ucs_status_t> requestStatus(2, UCS_INPROGRESS);

  auto checkStatus = [&requestStatus](ucs_status_t status, ::ucxx::RequestCallbackUserData data) {
    auto idx           = *std::static_pointer_cast<size_t>(data);
    requestStatus[idx] = status;
  };

  auto checkCompletion = [&requestStatus, this]() {
    std::vector<size_t> completed(2, 0);
    while (std::accumulate(completed.begin(), completed.end(), 0) != 2) {
      _progressWorker();
      std::transform(
        requestStatus.begin(), requestStatus.end(), completed.begin(), [](ucs_status_t status) {
          return status == UCS_INPROGRESS ? 0 : 1;
        });
    }
  };

  auto sendIndex = std::make_shared<size_t>(0u);
  auto recvIndex = std::make_shared<size_t>(1u);

  // Submit and wait for transfers to complete via callbacks; the shared_ptr is discarded
  // but the request is kept alive by the endpoint's inflight-request registry.
  std::shared_ptr<ucxx::Request> sendReq =
    _ep->tagSend(_sendPtr[0], _messageSize, ucxx::Tag{0}, false, checkStatus, sendIndex);
  std::shared_ptr<ucxx::Request> recvReq = _ep->tagRecv(
    _recvPtr[0], _messageSize, ucxx::Tag{0}, ucxx::TagMaskFull, false, checkStatus, recvIndex);
  checkCompletion();

  copyResults();

  for (const auto status : requestStatus)
    ASSERT_THAT(status, UCS_OK);

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryGet)
{
  allocate();

  auto memoryHandle = _context->createMemoryHandle(_messageSize, nullptr, _memoryType);
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  // Fill memory handle with send data
  copyMemoryTypeAware(
    reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _sendPtr[0], _messageSize);

  auto localRemoteKey      = memoryHandle->createRemoteKey();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->memGet(_recvPtr[0], _messageSize, remoteKey));
  requests.push_back(_ep->flush());
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryGetPreallocated)
{
  allocate();

  // Memory handles are always non-const
  auto memoryHandle =
    _context->createMemoryHandle(_messageSize, const_cast<void*>(_sendPtr[0]), _memoryType);
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  auto localRemoteKey      = memoryHandle->createRemoteKey();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->memGet(_recvPtr[0], _messageSize, remoteKey));
  requests.push_back(_ep->flush());
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryGetWithOffset)
{
  if (_messageLength < 2) GTEST_SKIP() << "Message too small to perform operations with offsets";
  allocate();

  size_t offset      = 1;
  size_t offsetBytes = offset * sizeof(_send[0][0]);

  auto memoryHandle = _context->createMemoryHandle(_messageSize, nullptr, _memoryType);
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  // Fill memory handle with send data
  copyMemoryTypeAware(
    reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _sendPtr[0], _messageSize);

  auto localRemoteKey      = memoryHandle->createRemoteKey();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->memGet(reinterpret_cast<char*>(_recvPtr[0]) + offsetBytes,
                                 _messageSize - offsetBytes,
                                 remoteKey,
                                 offsetBytes));
  requests.push_back(_ep->flush());
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  // Assert offset data correctness
  auto recvOffset = DataContainerType(_recv[0].begin() + offset, _recv[0].end());
  auto sendOffset = DataContainerType(_send[0].begin() + offset, _send[0].end());
  ASSERT_THAT(recvOffset, sendOffset);
}

TEST_P(RequestTest, MemoryPut)
{
  allocate();

  auto memoryHandle = _context->createMemoryHandle(_messageSize, nullptr, _memoryType);
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  auto localRemoteKey      = memoryHandle->createRemoteKey();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->memPut(_sendPtr[0], _messageSize, remoteKey));
  requests.push_back(_ep->flush());
  waitRequests(_worker, requests, _progressWorker);

  // Copy memory handle data to receive buffer
  copyMemoryTypeAware(
    _recvPtr[0], reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _messageSize);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryPutPreallocated)
{
  allocate();

  auto memoryHandle = _context->createMemoryHandle(_messageSize, _recvPtr[0], _memoryType);
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  auto localRemoteKey      = memoryHandle->createRemoteKey();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->memPut(_sendPtr[0], _messageSize, remoteKey));
  requests.push_back(_ep->flush());
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryPutWithOffset)
{
  if (_messageLength < 2) GTEST_SKIP() << "Message too small to perform operations with offsets";
  allocate();

  size_t offset      = 1;
  size_t offsetBytes = offset * sizeof(_send[0][0]);

  auto memoryHandle = _context->createMemoryHandle(_messageSize, nullptr, _memoryType);
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  auto localRemoteKey      = memoryHandle->createRemoteKey();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->memPut(reinterpret_cast<const char*>(_sendPtr[0]) + offsetBytes,
                                 _messageSize - offsetBytes,
                                 remoteKey,
                                 offsetBytes));
  requests.push_back(_ep->flush());
  waitRequests(_worker, requests, _progressWorker);

  // Copy memory handle data to receive buffer
  copyMemoryTypeAware(
    _recvPtr[0], reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _messageSize);

  copyResults();

  // Assert offset data correctness
  auto recvOffset = DataContainerType(_recv[0].begin() + offset, _recv[0].end());
  auto sendOffset = DataContainerType(_send[0].begin() + offset, _send[0].end());
  ASSERT_THAT(recvOffset, sendOffset);
}

// ---------------------------------------------------------------------------
// Active Message tests
// ---------------------------------------------------------------------------

// Helper: send a small eager message to AM handler ID, drive progress until the handler
// fires, then drain the send request. Does NOT call checkError() since UCX may propagate
// UCS_ERR_IO_ERROR from the handler back to the sender.
static void sendAndDrainAmEager(std::shared_ptr<ucxx::Endpoint> ep,
                                std::shared_ptr<ucxx::Worker> worker,
                                std::function<void()> progressWorker,
                                ucxx::AmHandlerId id,
                                std::atomic<bool>* handlerFired)
{
  std::vector<int> send{1, 2, 3};
  const size_t sendBytes = send.size() * sizeof(int);
  auto sendReq           = ep->amSend(id, nullptr, 0, ucxx::AmSendContig{send.data(), sendBytes});

  while (!handlerFired->load())
    progressWorker();
  while (!sendReq->isCompleted())
    progressWorker();
}

// Verify: the handler is invoked and the exception does not escape the AM receive callback.
// If the exception escaped into UCX's C stack the process would crash (std::terminate),
// so surviving to the end of the test is the proof the catch block ran.
TEST_P(RequestTest, ProgressAmHandlerExceptionIsCaught)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  std::atomic<bool> handlerFired{false};

  _worker->setAmHandler(8, [&handlerFired](ucxx::AmMessage&) {
    handlerFired.store(true);
    throw std::runtime_error("test exception from AM handler");
  });

  sendAndDrainAmEager(_ep, _worker, _progressWorker, 8, &handlerFired);
}

// Verify: when no onException callback is set, the AM receive callback logs a warning via
// ucxx_warn (which routes through UCX's log system).
TEST_P(RequestTest, ProgressAmHandlerExceptionLogsWarning)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  std::atomic<bool> handlerFired{false};
  gCapturedLogMessage[0] = '\0';
  gCapturedLogMessageSet.store(false, std::memory_order_release);
  ucs_log_push_handler(captureLogHandler);

  _worker->setAmHandler(9, [&handlerFired](ucxx::AmMessage&) {
    handlerFired.store(true);
    throw std::runtime_error("test exception for log check");
  });

  sendAndDrainAmEager(_ep, _worker, _progressWorker, 9, &handlerFired);

  const bool captured = loopWithTimeout(std::chrono::seconds(1), []() {
    return gCapturedLogMessageSet.load(std::memory_order_acquire);
  });
  ucs_log_pop_handler();

  EXPECT_TRUE(captured);
  EXPECT_THAT(std::string{gCapturedLogMessage},
              HasSubstr("Exception thrown in AM handler callback"));
}

TEST_P(RequestTest, ProgressAmEager)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  const size_t payloadLength = 16;
  std::vector<int> send(payloadLength);
  std::iota(send.begin(), send.end(), 42);
  const size_t payloadBytes = payloadLength * sizeof(int);

  std::vector<int> received(payloadLength, 0);
  std::atomic<bool> handlerFired{false};
  std::atomic<size_t> receivedLength{0};

  _worker->setAmHandler(1, [&received, &handlerFired, &receivedLength](ucxx::AmMessage& msg) {
    receivedLength.store(msg.length());
    if (msg.data()) std::memcpy(received.data(), msg.data(), msg.length());
    handlerFired.store(true);
  });

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSend(
    static_cast<uint16_t>(1), nullptr, 0, ucxx::AmSendContig{send.data(), payloadBytes}));
  waitRequests(_worker, requests, _progressWorker);

  while (!handlerFired.load())
    _progressWorker();

  EXPECT_EQ(receivedLength.load(), payloadBytes);
  ASSERT_THAT(received, ContainerEq(send));
}

TEST_P(RequestTest, ProgressAmRendezvous)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  // Force rendezvous by using a buffer larger than the threshold.
  const size_t payloadLength = _rndvThresh / sizeof(int) + 64;
  std::vector<int> send(payloadLength);
  std::iota(send.begin(), send.end(), 0);
  const size_t payloadBytes = payloadLength * sizeof(int);

  std::vector<int> received(payloadLength, 0);
  std::shared_ptr<ucxx::Request> recvReq;
  std::atomic<bool> handlerFired{false};

  _worker->setAmHandler(2,
                        [&received, &recvReq, &handlerFired, payloadBytes](ucxx::AmMessage& msg) {
                          if (!msg.isRendezvous()) return;
                          recvReq = msg.receive(received.data(), payloadBytes);
                          handlerFired.store(true);
                        });

  std::vector<std::shared_ptr<ucxx::Request>> sendRequests;
  sendRequests.push_back(_ep->amSend(
    static_cast<uint16_t>(2), nullptr, 0, ucxx::AmSendContig{send.data(), payloadBytes}));
  waitRequests(_worker, sendRequests, _progressWorker);

  while (!handlerFired.load())
    _progressWorker();

  std::vector<std::shared_ptr<ucxx::Request>> recvRequests{recvReq};
  waitRequests(_worker, recvRequests, _progressWorker);

  ASSERT_THAT(received, ContainerEq(send));
}

TEST_P(RequestTest, ProgressAmIov)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  const size_t payloadLength = 8;
  std::vector<int> send(payloadLength);
  std::iota(send.begin(), send.end(), 100);

  const size_t half = payloadLength / 2;
  std::vector<ucp_dt_iov_t> iov(2);
  iov[0].buffer = send.data();
  iov[0].length = half * sizeof(int);
  iov[1].buffer = send.data() + half;
  iov[1].length = (payloadLength - half) * sizeof(int);

  const size_t payloadBytes = payloadLength * sizeof(int);
  std::vector<int> received(payloadLength, 0);
  std::atomic<bool> handlerFired{false};
  std::atomic<size_t> receivedLength{0};

  _worker->setAmHandler(3, [&received, &handlerFired, &receivedLength](ucxx::AmMessage& msg) {
    receivedLength.store(msg.length());
    if (msg.data()) std::memcpy(received.data(), msg.data(), msg.length());
    handlerFired.store(true);
  });

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(
    _ep->amSend(static_cast<uint16_t>(3), nullptr, 0, ucxx::AmSendIov{std::move(iov)}));
  waitRequests(_worker, requests, _progressWorker);

  while (!handlerFired.load())
    _progressWorker();

  EXPECT_EQ(receivedLength.load(), payloadBytes);
  ASSERT_THAT(received, ContainerEq(send));
}

TEST_P(RequestTest, ProgressAmMultipleIds)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  std::vector<int> sendA{1, 2, 3, 4};
  std::vector<int> sendB{10, 20, 30, 40};
  const size_t sizeA = sendA.size() * sizeof(int);
  const size_t sizeB = sendB.size() * sizeof(int);

  std::vector<int> recvA(sendA.size(), 0);
  std::vector<int> recvB(sendB.size(), 0);
  std::atomic<int> fired{0};
  std::atomic<size_t> receivedLengthA{0};
  std::atomic<size_t> receivedLengthB{0};

  auto makeHandler = [&fired](std::vector<int>& dst, std::atomic<size_t>& receivedLen) {
    return [&dst, &fired, &receivedLen](ucxx::AmMessage& msg) {
      receivedLen.store(msg.length());
      if (msg.data()) std::memcpy(dst.data(), msg.data(), msg.length());
      fired.fetch_add(1);
    };
  };

  _worker->setAmHandler(4, makeHandler(recvA, receivedLengthA));
  _worker->setAmHandler(5, makeHandler(recvB, receivedLengthB));

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(
    _ep->amSend(static_cast<uint16_t>(4), nullptr, 0, ucxx::AmSendContig{sendA.data(), sizeA}));
  requests.push_back(
    _ep->amSend(static_cast<uint16_t>(5), nullptr, 0, ucxx::AmSendContig{sendB.data(), sizeB}));
  waitRequests(_worker, requests, _progressWorker);

  while (fired.load() < 2)
    _progressWorker();

  EXPECT_EQ(receivedLengthA.load(), sizeA);
  EXPECT_EQ(receivedLengthB.load(), sizeB);
  ASSERT_THAT(recvA, ContainerEq(sendA));
  ASSERT_THAT(recvB, ContainerEq(sendB));
}

TEST_P(RequestTest, ProgressAmAndManagedCoexist)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_memoryType == UCS_MEMORY_TYPE_CUDA) {
#if UCXX_ENABLE_CCCL
    _worker->registerManagedAmAllocator(UCS_MEMORY_TYPE_CUDA, [](size_t length) {
      return std::make_shared<ucxx::CCCLBuffer>(length);
    });
#else
    GTEST_SKIP() << "CCCL support is required for CUDA receive allocations";
#endif
  }

  allocate(1, false);

  // Register a handler on ID 1 for the thin API.
  const size_t thinPayloadLength = 4;
  std::vector<int> thinSend(thinPayloadLength);
  std::iota(thinSend.begin(), thinSend.end(), 200);
  const size_t thinBytes = thinPayloadLength * sizeof(int);

  std::vector<int> thinRecv(thinPayloadLength, 0);
  std::atomic<bool> thinHandlerFired{false};

  _worker->setAmHandler(6, [&thinRecv, &thinHandlerFired, thinBytes](ucxx::AmMessage& msg) {
    if (msg.data() && msg.length() == thinBytes)
      std::memcpy(thinRecv.data(), msg.data(), thinBytes);
    thinHandlerFired.store(true);
  });

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  // Thin API send on ID 6.
  requests.push_back(_ep->amSend(
    static_cast<uint16_t>(6), nullptr, 0, ucxx::AmSendContig{thinSend.data(), thinBytes}));
  // Managed API send/recv on ID 0.
  requests.push_back(_ep->amManagedSend(_sendPtr[0], _messageSize, _memoryType));
  requests.push_back(_ep->amManagedRecv());
  waitRequests(_worker, requests, _progressWorker);

  while (!thinHandlerFired.load())
    _progressWorker();

  auto managedRecvReq = requests[2];
  _recvPtr[0]         = managedRecvReq->getRecvBuffer()->data();
  copyResults();

  ASSERT_THAT(thinRecv, ContainerEq(thinSend));
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressAmRawHeader)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  // Binary header including null bytes and high bytes.
  const std::vector<uint8_t> sentHeader{0x00, 0x01, 0x42, 0xfe, 0xff};
  std::vector<int> sendData{7, 8, 9};
  const size_t dataBytes = sendData.size() * sizeof(int);

  std::vector<uint8_t> receivedHeader;
  std::vector<int> receivedData(sendData.size(), 0);
  std::atomic<bool> handlerFired{false};

  _worker->setAmHandler(
    7, [&receivedHeader, &receivedData, &handlerFired, dataBytes](ucxx::AmMessage& msg) {
      if (msg.header() && msg.headerLength() > 0)
        receivedHeader.assign(static_cast<const uint8_t*>(msg.header()),
                              static_cast<const uint8_t*>(msg.header()) + msg.headerLength());
      if (msg.data() && msg.length() == dataBytes)
        std::memcpy(receivedData.data(), msg.data(), dataBytes);
      handlerFired.store(true);
    });

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSend(static_cast<uint16_t>(7),
                                 sentHeader.data(),
                                 sentHeader.size(),
                                 ucxx::AmSendContig{sendData.data(), dataBytes}));
  waitRequests(_worker, requests, _progressWorker);

  while (!handlerFired.load())
    _progressWorker();

  ASSERT_EQ(receivedHeader, sentHeader);
  ASSERT_THAT(receivedData, ContainerEq(sendData));
}

INSTANTIATE_TEST_SUITE_P(ProgressModes,
                         RequestTest,
                         Combine(Values(TestBufferType::Host),
                                 Values(false),
                                 Values(false),
                                 Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        // ProgressMode::Wait,  // Hangs on Stream
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));

INSTANTIATE_TEST_SUITE_P(DelayedSubmission,
                         RequestTest,
                         Combine(Values(TestBufferType::Host),
                                 Values(false),
                                 Values(true),
                                 Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));

#if UCXX_TESTS_ENABLE_RMM
INSTANTIATE_TEST_SUITE_P(RMMProgressModes,
                         RequestTest,
                         Combine(Values(TestBufferType::RMM),
                                 Values(false, true),
                                 Values(false),
                                 Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        // ProgressMode::Wait,  // Hangs on Stream
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));

INSTANTIATE_TEST_SUITE_P(RMMDelayedSubmission,
                         RequestTest,
                         Combine(Values(TestBufferType::RMM),
                                 Values(false, true),
                                 Values(true),
                                 Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));
#endif

#if UCXX_ENABLE_CCCL
INSTANTIATE_TEST_SUITE_P(CCCLProgressModes,
                         RequestTest,
                         Combine(Values(TestBufferType::CCCL),
                                 Values(false, true),
                                 Values(false),
                                 Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        // ProgressMode::Wait,  // Hangs on Stream
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));

INSTANTIATE_TEST_SUITE_P(CCCLDelayedSubmission,
                         RequestTest,
                         Combine(Values(TestBufferType::CCCL),
                                 Values(false, true),
                                 Values(true),
                                 Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));
#endif

}  // namespace
