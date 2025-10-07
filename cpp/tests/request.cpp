/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "include/utils.h"
#include "ucxx/buffer.h"
#include "ucxx/constructors.h"
#include "ucxx/utils/ucx.h"

#if UCXX_ENABLE_RMM
#include <rmm/device_buffer.hpp>
#endif

namespace {

using ::testing::Combine;
using ::testing::ContainerEq;
using ::testing::Values;

typedef std::vector<int> DataContainerType;

class RequestTest : public ::testing::TestWithParam<
                      std::tuple<ucxx::BufferType, bool, bool, ProgressMode, size_t>> {
 protected:
  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _ep{nullptr};
  std::function<void()> _progressWorker;

  ucxx::BufferType _bufferType;
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

  void SetUp()
  {
    std::tie(_bufferType,
             _registerCustomAmAllocator,
             _enableDelayedSubmission,
             _progressMode,
             _messageLength) = GetParam();

    if (_bufferType == ucxx::BufferType::RMM) {
#if !UCXX_ENABLE_RMM
      GTEST_SKIP() << "UCXX was not built with RMM support";
#endif
    }

    _memoryType =
      (_bufferType == ucxx::BufferType::RMM) ? UCS_MEMORY_TYPE_CUDA : UCS_MEMORY_TYPE_HOST;
    _messageSize = _messageLength * sizeof(int);

    _context = ucxx::createContext({{"RNDV_THRESH", std::to_string(_rndvThresh)}},
                                   ucxx::Context::defaultFeatureFlags);
    _worker  = _context->createWorker(_enableDelayedSubmission);

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

      if (_bufferType == ucxx::BufferType::Host) {
        _sendBuffer[i] = std::make_unique<ucxx::HostBuffer>(_messageSize);
        if (allocateRecvBuffer) _recvBuffer[i] = std::make_unique<ucxx::HostBuffer>(_messageSize);
#if UCXX_ENABLE_RMM
      } else if (_bufferType == ucxx::BufferType::RMM) {
        _sendBuffer[i] = std::make_unique<ucxx::RMMBuffer>(_messageSize);
        if (allocateRecvBuffer) _recvBuffer[i] = std::make_unique<ucxx::RMMBuffer>(_messageSize);
#endif
      }

      copyMemoryTypeAware(_sendBuffer[i]->data(), _send[i].data(), _messageSize, false);

      _sendPtr[i] = _sendBuffer[i]->data();
      if (allocateRecvBuffer) _recvPtr[i] = _recvBuffer[i]->data();
    }
#if UCXX_ENABLE_RMM
    if (_bufferType == ucxx::BufferType::RMM) { rmm::cuda_stream_default.synchronize(); }
#endif
  }

  void copyResults()
  {
    for (size_t i = 0; i < _numBuffers; ++i)
      copyMemoryTypeAware(_recv[i].data(), _recvPtr[i], _messageSize, false);
#if UCXX_ENABLE_RMM
    if (_bufferType == ucxx::BufferType::RMM) { rmm::cuda_stream_default.synchronize(); }
#endif
  }

  void copyMemoryTypeAware(void* dst, const void* src, size_t size, bool synchronize = true)
  {
    if (_memoryType == UCS_MEMORY_TYPE_HOST) {
      memcpy(dst, src, size);
#if UCXX_ENABLE_RMM
    } else if (_memoryType == UCS_MEMORY_TYPE_CUDA) {
      RMM_CUDA_TRY(
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, rmm::cuda_stream_default.value()));
      if (synchronize) rmm::cuda_stream_default.synchronize();
#endif
    }
  }
};

TEST_P(RequestTest, ProgressAm)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_registerCustomAmAllocator && _memoryType == UCS_MEMORY_TYPE_CUDA) {
#if !UCXX_ENABLE_RMM
    GTEST_SKIP() << "UCXX was not built with RMM support";
#else
    _worker->registerAmAllocator(UCS_MEMORY_TYPE_CUDA, [](size_t length) {
      return std::make_shared<ucxx::RMMBuffer>(length);
    });
#endif
  }

  allocate(1, false);

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSend(_sendPtr[0], _messageSize, _memoryType));
  requests.push_back(_ep->amRecv());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq = requests[1];
  _recvPtr[0]  = recvReq->getRecvBuffer()->data();

  // Messages larger than `_rndvThresh` are rendezvous and will use custom allocator,
  // smaller messages are eager and will always be host-allocated.
  ASSERT_THAT(recvReq->getRecvBuffer()->getType(),
              (_registerCustomAmAllocator && _messageSize >= _rndvThresh) ? _bufferType
                                                                          : ucxx::BufferType::Host);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressAmReceiverCallback)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_registerCustomAmAllocator && _memoryType == UCS_MEMORY_TYPE_CUDA) {
#if !UCXX_ENABLE_RMM
    GTEST_SKIP() << "UCXX was not built with RMM support";
#else
    _worker->registerAmAllocator(UCS_MEMORY_TYPE_CUDA, [](size_t length) {
      return std::make_shared<ucxx::RMMBuffer>(length);
    });
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
  _worker->registerAmReceiverCallback(receiverCallbackInfo, callback);

  allocate(1, false);

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSend(_sendPtr[0], _messageSize, _memoryType, receiverCallbackInfo));
  waitRequests(_worker, requests, _progressWorker);

  while (receivedRequests.size() < 1)
    _progressWorker();

  {
    std::lock_guard<std::mutex> lock(mutex);
    _recvPtr[0] = receivedRequests[0]->getRecvBuffer()->data();

    // Messages larger than `_rndvThresh` are rendezvous and will use custom allocator,
    // smaller messages are eager and will always be host-allocated.
    ASSERT_THAT(receivedRequests[0]->getRecvBuffer()->getType(),
                (_registerCustomAmAllocator && _messageSize >= _rndvThresh)
                  ? _bufferType
                  : ucxx::BufferType::Host);
  }

  copyResults();

  // Assert data correctness
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

TEST_P(RequestTest, ProgressTagMulti)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  const size_t numMulti         = 8;
  const bool allocateRecvBuffer = false;

  allocate(numMulti, allocateRecvBuffer);

  // Allocate buffers for request sizes/types
  std::vector<size_t> multiSize(numMulti, _messageSize);
  std::vector<int> multiIsCUDA(numMulti, _bufferType == ucxx::BufferType::RMM);

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
      ASSERT_EQ(br->buffer->getType(), _bufferType);
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

  // Submit and wait for transfers to complete
  std::ignore =
    _ep->tagSend(_sendPtr[0], _messageSize, ucxx::Tag{0}, false, checkStatus, sendIndex);
  std::ignore = _ep->tagRecv(
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

INSTANTIATE_TEST_SUITE_P(ProgressModes,
                         RequestTest,
                         Combine(Values(ucxx::BufferType::Host),
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
                         Combine(Values(ucxx::BufferType::Host),
                                 Values(false),
                                 Values(true),
                                 Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));

#if UCXX_ENABLE_RMM
INSTANTIATE_TEST_SUITE_P(RMMProgressModes,
                         RequestTest,
                         Combine(Values(ucxx::BufferType::RMM),
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
                         Combine(Values(ucxx::BufferType::RMM),
                                 Values(false, true),
                                 Values(true),
                                 Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));
#endif

}  // namespace
