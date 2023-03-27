/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "utils.h"

namespace {

using ::testing::Combine;
using ::testing::ContainerEq;
using ::testing::Values;

class RequestTest
  : public ::testing::TestWithParam<std::tuple<ucxx::BufferType, bool, ProgressMode, size_t>> {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::createContext({}, ucxx::Context::defaultFeatureFlags)};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _ep{nullptr};
  std::function<void()> _progressWorker;

  ucxx::BufferType _bufferType;
  bool _enableDelayedSubmission;
  ProgressMode _progressMode;
  size_t _messageLength;
  size_t _messageSize;

  size_t _numBuffers{0};
  std::vector<std::vector<int>> _send;
  std::vector<std::vector<int>> _recv;
  std::vector<std::unique_ptr<ucxx::Buffer>> _sendBuffer;
  std::vector<std::unique_ptr<ucxx::Buffer>> _recvBuffer;
  std::vector<void*> _sendPtr{nullptr};
  std::vector<void*> _recvPtr{nullptr};

  void SetUp()
  {
    if (_bufferType == ucxx::BufferType::RMM) {
#if !UCXX_ENABLE_RMM
      GTEST_SKIP() << "UCXX was not built with RMM support";
#endif
    }

    std::tie(_bufferType, _enableDelayedSubmission, _progressMode, _messageLength) = GetParam();
    _messageSize = _messageLength * sizeof(int);

    _worker = _context->createWorker(_enableDelayedSubmission);

    if (_progressMode == ProgressMode::Blocking)
      _worker->initBlockingProgressMode();
    else if (_progressMode == ProgressMode::ThreadPolling ||
             _progressMode == ProgressMode::ThreadBlocking) {
      _worker->setProgressThreadStartCallback(::createCudaContextCallback, nullptr);

      if (_progressMode == ProgressMode::ThreadPolling) _worker->startProgressThread(true);
      if (_progressMode == ProgressMode::ThreadBlocking) _worker->startProgressThread(false);
    }

    _progressWorker = getProgressFunction(_worker, _progressMode);

    _ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
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

        std::copy(_send[i].begin(), _send[i].end(), (int*)_sendBuffer[i]->data());
      }
#if UCXX_ENABLE_RMM
      else if (_bufferType == ucxx::BufferType::RMM) {
        _sendBuffer[i] = std::make_unique<ucxx::RMMBuffer>(_messageSize);
        if (allocateRecvBuffer) _recvBuffer[i] = std::make_unique<ucxx::RMMBuffer>(_messageSize);

        RMM_CUDA_TRY(cudaMemcpyAsync(_sendBuffer[i]->data(),
                                     _send[i].data(),
                                     _messageSize,
                                     cudaMemcpyDefault,
                                     rmm::cuda_stream_default.value()));
      }
#endif

      _sendPtr[i] = _sendBuffer[i]->data();
      if (allocateRecvBuffer) _recvPtr[i] = _recvBuffer[i]->data();
    }
  }

  void copyResults()
  {
    for (size_t i = 0; i < _numBuffers; ++i) {
      if (_bufferType == ucxx::BufferType::Host) {
        std::copy((int*)_recvPtr[i], (int*)_recvPtr[i] + _messageLength, _recv[i].begin());
      }
#if UCXX_ENABLE_RMM
      else if (_bufferType == ucxx::BufferType::RMM) {
        RMM_CUDA_TRY(cudaMemcpyAsync(_recv[i].data(),
                                     _recvPtr[i],
                                     _messageSize,
                                     cudaMemcpyDefault,
                                     rmm::cuda_stream_default.value()));
      }
#endif
    }
  }
};

TEST_P(RequestTest, ProgressStream)
{
  allocate();

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->streamSend(_sendPtr[0], _messageSize, 0));
  requests.push_back(_ep->streamRecv(_recvPtr[0], _messageSize, 0));
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressTag)
{
  allocate();

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->tagSend(_sendPtr[0], _messageSize, 0));
  requests.push_back(_ep->tagRecv(_recvPtr[0], _messageSize, 0));
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
  std::vector<std::shared_ptr<ucxx::RequestTagMulti>> requests;
  requests.push_back(_ep->tagMultiSend(_sendPtr, multiSize, multiIsCUDA, 0, false));
  requests.push_back(_ep->tagMultiRecv(0, false));
  waitRequestsTagMulti(_worker, requests, _progressWorker);

  auto recvRequest = requests[1];

  _recvPtr.resize(_numBuffers);
  size_t transferIdx = 0;

  // Populate recv pointers
  for (const auto& br : recvRequest->_bufferRequests) {
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

INSTANTIATE_TEST_SUITE_P(ProgressModes,
                         RequestTest,
                         Combine(Values(ucxx::BufferType::Host),
                                 Values(false),
                                 Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        // ProgressMode::Wait,  // Hangs on Stream
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking),
                                 Values(1, 1024, 1048576)));

INSTANTIATE_TEST_SUITE_P(DelayedSubmission,
                         RequestTest,
                         Combine(Values(ucxx::BufferType::Host),
                                 Values(true),
                                 Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
                                 Values(1, 1024, 1048576)));

#if UCXX_ENABLE_RMM
INSTANTIATE_TEST_SUITE_P(RMMProgressModes,
                         RequestTest,
                         Combine(Values(ucxx::BufferType::RMM),
                                 Values(false),
                                 Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        // ProgressMode::Wait,  // Hangs on Stream
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking),
                                 Values(1, 1024, 1048576)));

INSTANTIATE_TEST_SUITE_P(RMMDelayedSubmission,
                         RequestTest,
                         Combine(Values(ucxx::BufferType::RMM),
                                 Values(true),
                                 Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
                                 Values(1, 1024, 1048576)));
#endif

}  // namespace
