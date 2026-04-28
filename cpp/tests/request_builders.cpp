/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <ucxx/api.h>
#include <ucxx/request_am.h>
#include <ucxx/request_endpoint_close.h>
#include <ucxx/request_flush.h>
#include <ucxx/request_mem.h>
#include <ucxx/request_stream.h>
#include <ucxx/request_tag.h>

#include "include/utils.h"

namespace {

class RequestBuilderTest : public ::testing::Test {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::createContext({}, ucxx::Context::defaultFeatureFlags)};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _ep{nullptr};

  void SetUp() override
  {
    _worker = _context->createWorker();
    _ep     = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  }

  void progressUntilCompleted(std::shared_ptr<ucxx::Request> req)
  {
    while (!req->isCompleted())
      _worker->progress();
    req->checkError();
  }
};

// ============================================================================
// RequestFlushBuilder tests
// ============================================================================

TEST_F(RequestBuilderTest, FlushBuilderBasicWorker)
{
  auto req = ucxx::experimental::createRequestFlush(_worker, ucxx::data::Flush{}).build();

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, FlushBuilderBasicEndpoint)
{
  auto req = ucxx::experimental::createRequestFlush(_ep, ucxx::data::Flush{}).build();

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, FlushBuilderWithPythonFuture)
{
  auto req = ucxx::experimental::createRequestFlush(_worker, ucxx::data::Flush{})
               .pythonFuture(false)
               .build();

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, FlushBuilderMethodChaining)
{
  ucxx::RequestCallbackUserFunction callbackFn{nullptr};
  ucxx::RequestCallbackUserData callbackData{nullptr};
  auto req = ucxx::experimental::createRequestFlush(_worker, ucxx::data::Flush{})
               .pythonFuture(false)
               .callbackFunction(callbackFn)
               .callbackData(callbackData)
               .build();

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, FlushBuilderImplicitConversion)
{
  std::shared_ptr<ucxx::RequestFlush> req =
    ucxx::experimental::createRequestFlush(_worker, ucxx::data::Flush{});

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

// ============================================================================
// RequestTagBuilder tests
// ============================================================================

TEST_F(RequestBuilderTest, TagBuilderSendReceivePair)
{
  std::vector<int> sendBuf{42, 43, 44};
  std::vector<int> recvBuf(3);
  auto tag = ucxx::Tag{1};

  auto sendData = ucxx::data::TagSend(sendBuf.data(), sendBuf.size() * sizeof(int), tag);
  auto recvData =
    ucxx::data::TagReceive(recvBuf.data(), recvBuf.size() * sizeof(int), tag, ucxx::TagMaskFull);

  auto sendReq = ucxx::experimental::createRequestTag(_ep, sendData).build();
  auto recvReq = ucxx::experimental::createRequestTag(_worker, recvData).build();

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();

  EXPECT_EQ(sendBuf, recvBuf);
}

TEST_F(RequestBuilderTest, TagBuilderWithPythonFuture)
{
  std::vector<int> sendBuf{1};
  std::vector<int> recvBuf(1);
  auto tag = ucxx::Tag{2};

  auto sendData = ucxx::data::TagSend(sendBuf.data(), sendBuf.size() * sizeof(int), tag);
  auto recvData =
    ucxx::data::TagReceive(recvBuf.data(), recvBuf.size() * sizeof(int), tag, ucxx::TagMaskFull);

  auto sendReq = ucxx::experimental::createRequestTag(_ep, sendData).pythonFuture(false).build();
  auto recvReq =
    ucxx::experimental::createRequestTag(_worker, recvData).pythonFuture(false).build();

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();
}

TEST_F(RequestBuilderTest, TagBuilderImplicitConversion)
{
  std::vector<int> sendBuf{7};
  std::vector<int> recvBuf(1);
  auto tag = ucxx::Tag{3};

  auto sendData = ucxx::data::TagSend(sendBuf.data(), sendBuf.size() * sizeof(int), tag);
  auto recvData =
    ucxx::data::TagReceive(recvBuf.data(), recvBuf.size() * sizeof(int), tag, ucxx::TagMaskFull);

  std::shared_ptr<ucxx::RequestTag> sendReq = ucxx::experimental::createRequestTag(_ep, sendData);
  std::shared_ptr<ucxx::RequestTag> recvReq =
    ucxx::experimental::createRequestTag(_worker, recvData);

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();
}

// ============================================================================
// RequestStreamBuilder tests
// ============================================================================

TEST_F(RequestBuilderTest, StreamBuilderSendReceivePair)
{
  std::vector<int> sendBuf{10, 20};
  std::vector<int> recvBuf(2);

  auto sendData = ucxx::data::StreamSend(sendBuf.data(), sendBuf.size() * sizeof(int));
  auto recvData = ucxx::data::StreamReceive(recvBuf.data(), recvBuf.size() * sizeof(int));
  auto sendReq  = ucxx::experimental::createRequestStream(_ep, sendData).build();
  auto recvReq  = ucxx::experimental::createRequestStream(_ep, recvData).build();

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();
}

TEST_F(RequestBuilderTest, StreamBuilderImplicitConversion)
{
  std::vector<int> sendBuf{99};
  std::vector<int> recvBuf(1);

  auto sendData = ucxx::data::StreamSend(sendBuf.data(), sendBuf.size() * sizeof(int));
  auto recvData = ucxx::data::StreamReceive(recvBuf.data(), recvBuf.size() * sizeof(int));

  std::shared_ptr<ucxx::RequestStream> sendReq =
    ucxx::experimental::createRequestStream(_ep, sendData);
  std::shared_ptr<ucxx::RequestStream> recvReq =
    ucxx::experimental::createRequestStream(_ep, recvData);

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();
}

// ============================================================================
// RequestEndpointCloseBuilder tests
// ============================================================================

// Note: createRequestEndpointClose bypasses the Endpoint::_closing flag, so calling
// it directly (outside of Endpoint::close()) risks a double-close when the Endpoint
// destructs. Builder type correctness is verified via static_assert in AllBuilderAutoTypes.

TEST_F(RequestBuilderTest, EndpointCloseBuilderAutoType)
{
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  auto builder =
    ucxx::experimental::createRequestEndpointClose(ep, ucxx::data::EndpointClose{false});
  static_assert(
    std::is_same<decltype(builder), ucxx::experimental::RequestEndpointCloseBuilder>::value,
    "auto without .build() is RequestEndpointCloseBuilder");
}

TEST_F(RequestBuilderTest, EndpointCloseBuilderMethodChaining)
{
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());

  ucxx::RequestCallbackUserFunction callbackFn{nullptr};
  ucxx::RequestCallbackUserData callbackData{nullptr};
  auto builder =
    ucxx::experimental::createRequestEndpointClose(ep, ucxx::data::EndpointClose{false})
      .pythonFuture(false)
      .callbackFunction(callbackFn)
      .callbackData(callbackData);

  static_assert(
    std::is_same<decltype(builder), ucxx::experimental::RequestEndpointCloseBuilder>::value,
    "chained builder without .build() is still RequestEndpointCloseBuilder");
}

// ============================================================================
// Static type assertion tests for all builders
// ============================================================================

TEST_F(RequestBuilderTest, AllBuilderAutoTypes)
{
  std::vector<int> buf{0};
  auto tag = ucxx::Tag{0};

  // RequestTagBuilder
  auto tagBuilder =
    ucxx::experimental::createRequestTag(_ep, ucxx::data::TagSend(buf.data(), sizeof(int), tag));
  static_assert(std::is_same<decltype(tagBuilder), ucxx::experimental::RequestTagBuilder>::value,
                "auto without .build() is RequestTagBuilder");

  // RequestAmBuilder
  auto amBuilder =
    ucxx::experimental::createRequestAm(_ep, ucxx::data::AmSend(buf.data(), sizeof(int)));
  static_assert(std::is_same<decltype(amBuilder), ucxx::experimental::RequestAmBuilder>::value,
                "auto without .build() is RequestAmBuilder");

  // RequestMemBuilder - just test the builder type, don't build (needs real rkey)
  // (ucp_rkey_h cannot be constructed in tests without remote memory registration)

  // RequestStreamBuilder
  auto streamBuilder =
    ucxx::experimental::createRequestStream(_ep, ucxx::data::StreamSend(buf.data(), sizeof(int)));
  static_assert(
    std::is_same<decltype(streamBuilder), ucxx::experimental::RequestStreamBuilder>::value,
    "auto without .build() is RequestStreamBuilder");

  // RequestFlushBuilder
  auto flushBuilder = ucxx::experimental::createRequestFlush(_worker, ucxx::data::Flush{});
  static_assert(
    std::is_same<decltype(flushBuilder), ucxx::experimental::RequestFlushBuilder>::value,
    "auto without .build() is RequestFlushBuilder");

  // RequestEndpointCloseBuilder (using a temporary endpoint to avoid double-close on _ep)
  {
    auto closeEp = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
    auto closeBuilder =
      ucxx::experimental::createRequestEndpointClose(closeEp, ucxx::data::EndpointClose{false});
    static_assert(
      std::is_same<decltype(closeBuilder), ucxx::experimental::RequestEndpointCloseBuilder>::value,
      "auto without .build() is RequestEndpointCloseBuilder");
  }

  // RequestTagMultiBuilder
  std::vector<const void*> multiSendBuf{buf.data()};
  std::vector<size_t> multiSendLen{sizeof(int)};
  std::vector<int> isCUDA{0};
  auto tagMultiBuilder = ucxx::experimental::createRequestTagMulti(
    _ep, ucxx::data::TagMultiSend(multiSendBuf, multiSendLen, isCUDA, tag));
  static_assert(
    std::is_same<decltype(tagMultiBuilder), ucxx::experimental::RequestTagMultiBuilder>::value,
    "auto without .build() is RequestTagMultiBuilder");

  // Verify .build() types
  auto flushReq = flushBuilder.build();
  static_assert(std::is_same<decltype(flushReq), std::shared_ptr<ucxx::RequestFlush>>::value,
                "calling .build() on RequestFlushBuilder returns shared_ptr<RequestFlush>");

  ASSERT_TRUE(flushReq != nullptr);
  progressUntilCompleted(flushReq);
}

}  // namespace
