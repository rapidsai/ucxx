/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>
#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>
#include <ucs/type/status.h>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "include/utils.h"
#include "ucxx/buffer.h"
#include "ucxx/constructors.h"
#include "ucxx/utils/ucx.h"

namespace {

using ::testing::Combine;
using ::testing::Values;

class RmaTest : public ::testing::TestWithParam<std::tuple<ucs_memory_type_t, size_t, bool>> {
 protected:
  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _ep{nullptr};

  ucs_memory_type_t _memoryType;
  size_t _messageSize;
  bool _preallocateBuffer;
  size_t _rndvThresh{8192};
  void* _buffer{nullptr};

  void SetUp()
  {
    std::tie(_memoryType, _messageSize, _preallocateBuffer) = GetParam();

    _context = ucxx::createContext({{"RNDV_THRESH", std::to_string(_rndvThresh)}},
                                   ucxx::Context::defaultFeatureFlags);
    _worker  = _context->createWorker();

    _ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

    _buffer = preallocate();
  }

  void TearDown() { release(); }

  void* preallocate()
  {
    if (_preallocateBuffer) {
      if (_memoryType == UCS_MEMORY_TYPE_HOST)
        return malloc(_messageSize);
      else
        throw std::runtime_error("Unsupported memory type");
    }
    return nullptr;
  }

  void release()
  {
    if (_preallocateBuffer && _buffer != nullptr) {
      if (_memoryType == UCS_MEMORY_TYPE_HOST) free(_buffer);
    }
  }
};

class BasicUcxxRmaTest : public ::testing::TestWithParam<std::tuple<size_t>> {
 protected:
  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _ep{nullptr};

  size_t _messageSize;

  void SetUp()
  {
    std::tie(_messageSize) = GetParam();

    _context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
    _worker  = _context->createWorker();
    _ep      = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  }
};

TEST_P(RmaTest, MemoryHandle)
{
  auto memoryHandle = _context->createMemoryHandle(_messageSize, _buffer);
  ASSERT_GE(memoryHandle->getSize(), _messageSize);
  if (_messageSize == 0)
    ASSERT_EQ(memoryHandle->getBaseAddress(), 0);
  else
    ASSERT_NE(memoryHandle->getBaseAddress(), 0);

  ASSERT_NE(memoryHandle->getHandle(), nullptr);
}

TEST_P(RmaTest, MemoryHandleUcxxNamespaceConstructor)
{
  auto memoryHandle = ucxx::createMemoryHandle(_context, _messageSize, _buffer);
  ASSERT_GE(memoryHandle->getSize(), _messageSize);
  if (_messageSize == 0)
    ASSERT_EQ(memoryHandle->getBaseAddress(), 0);
  else
    ASSERT_NE(memoryHandle->getBaseAddress(), 0);

  ASSERT_NE(memoryHandle->getHandle(), nullptr);
}

TEST_P(RmaTest, RemoteKey)
{
  auto memoryHandle = _context->createMemoryHandle(_messageSize, _buffer);

  auto remoteKey = memoryHandle->createRemoteKey();

  ASSERT_EQ(remoteKey->getSize(), memoryHandle->getSize());
  ASSERT_EQ(remoteKey->getBaseAddress(), memoryHandle->getBaseAddress());
  ASSERT_EQ(remoteKey->getHandle(), nullptr);
}

TEST_P(RmaTest, RemoteKeyUcxxNamespaceConstructor)
{
  auto memoryHandle = ucxx::createMemoryHandle(_context, _messageSize, _buffer);

  auto remoteKey = ucxx::createRemoteKeyFromMemoryHandle(memoryHandle);

  ASSERT_EQ(remoteKey->getSize(), memoryHandle->getSize());
  ASSERT_EQ(remoteKey->getBaseAddress(), memoryHandle->getBaseAddress());
  ASSERT_EQ(remoteKey->getHandle(), nullptr);
}

TEST_P(RmaTest, RemoteKeySerialization)
{
  auto memoryHandle = _context->createMemoryHandle(_messageSize, _buffer);

  auto remoteKey = memoryHandle->createRemoteKey();

  auto serializedRemoteKey = remoteKey->serialize();

  auto deserializedRemoteKey = ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey);

  ASSERT_EQ(remoteKey->getSize(), deserializedRemoteKey->getSize());
  ASSERT_EQ(remoteKey->getBaseAddress(), deserializedRemoteKey->getBaseAddress());
  ASSERT_NE(deserializedRemoteKey->getHandle(), nullptr);
}

TEST_P(BasicUcxxRmaTest, RemoteKeyCorruptedSerializedData)
{
  auto memoryHandle = _context->createMemoryHandle(_messageSize, nullptr);

  auto remoteKey = memoryHandle->createRemoteKey();

  auto serializedRemoteKey = remoteKey->serialize();
  serializedRemoteKey[1]   = ~serializedRemoteKey[1];

  EXPECT_THROW(ucxx::createRemoteKeyFromSerialized(_ep, serializedRemoteKey), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(AttributeTests,
                         RmaTest,
                         Combine(Values(UCS_MEMORY_TYPE_HOST),
                                 Values(0, 1, 4, 4096, 8192, 4194304),
                                 Values(false, true)));

INSTANTIATE_TEST_SUITE_P(FailureTests, BasicUcxxRmaTest, Combine(Values(0, 4194304)));

}  // namespace
