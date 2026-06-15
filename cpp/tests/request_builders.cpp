/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <tuple>
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

template <typename BuilderType, typename TargetType>
void assertBuildRequiresNonConstBuilder()
{
  static_assert(std::is_invocable_r<std::shared_ptr<TargetType>,
                                    decltype(&BuilderType::build),
                                    BuilderType&>::value,
                "Non-const builder can call build() to produce target shared_ptr");
  static_assert(!std::is_invocable_r<std::shared_ptr<TargetType>,
                                     decltype(&BuilderType::build),
                                     const BuilderType&>::value,
                "Const builder cannot call build() to produce target shared_ptr");
}

template <typename BuilderType, typename TargetType>
void assertConversionRequiresNonConstBuilder()
{
  static_assert(std::is_convertible<BuilderType&, std::shared_ptr<TargetType>>::value,
                "Non-const builder can implicitly convert to target shared_ptr");
  static_assert(!std::is_convertible<const BuilderType&, std::shared_ptr<TargetType>>::value,
                "Const builder cannot implicitly convert to target shared_ptr");
}

TEST_F(RequestBuilderTest, FlushBuilderBasicWorker)
{
  auto req = ucxx::experimental::createRequestFlush(_worker, ucxx::data::Flush{}).build();
  static_assert(std::is_same<decltype(req), std::shared_ptr<ucxx::RequestFlush>>::value,
                ".build() returns shared_ptr<RequestFlush>");

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
  static_assert(std::is_same<decltype(sendReq), std::shared_ptr<ucxx::RequestTag>>::value,
                ".build() returns shared_ptr<RequestTag>");

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

TEST_F(RequestBuilderTest, StreamBuilderSendReceivePair)
{
  std::vector<int> sendBuf{10, 20};
  std::vector<int> recvBuf(2);

  auto sendData = ucxx::data::StreamSend(sendBuf.data(), sendBuf.size() * sizeof(int));
  auto recvData = ucxx::data::StreamReceive(recvBuf.data(), recvBuf.size() * sizeof(int));
  auto sendReq  = ucxx::experimental::createRequestStream(_ep, sendData).build();
  auto recvReq  = ucxx::experimental::createRequestStream(_ep, recvData).build();
  static_assert(std::is_same<decltype(sendReq), std::shared_ptr<ucxx::RequestStream>>::value,
                ".build() returns shared_ptr<RequestStream>");

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

TEST_F(RequestBuilderTest, EndpointCloseBuilderBuild)
{
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  auto req =
    ucxx::experimental::createRequestEndpointClose(ep, ucxx::data::EndpointClose{true}).build();

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
  EXPECT_EQ(nullptr, ep->close());
}

TEST_F(RequestBuilderTest, AllBuilderAutoTypes)
{
  std::vector<int> buf{0};
  auto tag = ucxx::Tag{0};

  auto tagBuilder =
    ucxx::experimental::createRequestTag(_ep, ucxx::data::TagSend(buf.data(), sizeof(int), tag));
  static_assert(std::is_same<decltype(tagBuilder), ucxx::experimental::RequestTagBuilder>::value,
                "auto without .build() is RequestTagBuilder");

  auto amBuilder =
    ucxx::experimental::createRequestAm(_ep, ucxx::data::AmSend(buf.data(), sizeof(int)));
  static_assert(std::is_same<decltype(amBuilder), ucxx::experimental::RequestAmBuilder>::value,
                "auto without .build() is RequestAmBuilder");

  // RequestMemBuilder - don't submit a request here; that needs a real remote key.
  auto memBuilder = ucxx::experimental::createRequestMem(
    _ep, ucxx::data::MemPut(buf.data(), sizeof(int), 0, nullptr));
  static_assert(std::is_same<decltype(memBuilder), ucxx::experimental::RequestMemBuilder>::value,
                "auto without .build() is RequestMemBuilder");
  static_assert(
    std::is_same<decltype(memBuilder.build()), std::shared_ptr<ucxx::RequestMem>>::value,
    "calling .build() on RequestMemBuilder returns shared_ptr<RequestMem>");

  auto streamBuilder =
    ucxx::experimental::createRequestStream(_ep, ucxx::data::StreamSend(buf.data(), sizeof(int)));
  static_assert(
    std::is_same<decltype(streamBuilder), ucxx::experimental::RequestStreamBuilder>::value,
    "auto without .build() is RequestStreamBuilder");

  auto flushBuilder = ucxx::experimental::createRequestFlush(_worker, ucxx::data::Flush{});
  static_assert(
    std::is_same<decltype(flushBuilder), ucxx::experimental::RequestFlushBuilder>::value,
    "auto without .build() is RequestFlushBuilder");

  {
    auto closeEp = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
    auto closeBuilder =
      ucxx::experimental::createRequestEndpointClose(closeEp, ucxx::data::EndpointClose{false});
    static_assert(
      std::is_same<decltype(closeBuilder), ucxx::experimental::RequestEndpointCloseBuilder>::value,
      "auto without .build() is RequestEndpointCloseBuilder");
  }

  std::vector<const void*> multiSendBuf{buf.data()};
  std::vector<size_t> multiSendLen{sizeof(int)};
  std::vector<int> isCUDA{0};
  auto tagMultiBuilder = ucxx::experimental::createRequestTagMulti(
    _ep, ucxx::data::TagMultiSend(multiSendBuf, multiSendLen, isCUDA, tag));
  static_assert(
    std::is_same<decltype(tagMultiBuilder), ucxx::experimental::RequestTagMultiBuilder>::value,
    "auto without .build() is RequestTagMultiBuilder");

  auto flushReq = flushBuilder.build();
  static_assert(std::is_same<decltype(flushReq), std::shared_ptr<ucxx::RequestFlush>>::value,
                "calling .build() on RequestFlushBuilder returns shared_ptr<RequestFlush>");

  ASSERT_TRUE(flushReq != nullptr);
  progressUntilCompleted(flushReq);
}

TEST(RequestBuilderTraitsTest, AllBuildersAreMoveOnly)
{
  static_assert(!std::is_copy_constructible<ucxx::experimental::RequestAmBuilder>::value,
                "RequestAmBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::experimental::RequestAmBuilder>::value,
                "RequestAmBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::experimental::RequestAmBuilder>::value,
                "RequestAmBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::experimental::RequestEndpointCloseBuilder>::value,
                "RequestEndpointCloseBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::experimental::RequestEndpointCloseBuilder>::value,
                "RequestEndpointCloseBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::experimental::RequestEndpointCloseBuilder>::value,
                "RequestEndpointCloseBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::experimental::RequestFlushBuilder>::value,
                "RequestFlushBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::experimental::RequestFlushBuilder>::value,
                "RequestFlushBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::experimental::RequestFlushBuilder>::value,
                "RequestFlushBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::experimental::RequestMemBuilder>::value,
                "RequestMemBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::experimental::RequestMemBuilder>::value,
                "RequestMemBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::experimental::RequestMemBuilder>::value,
                "RequestMemBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::experimental::RequestStreamBuilder>::value,
                "RequestStreamBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::experimental::RequestStreamBuilder>::value,
                "RequestStreamBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::experimental::RequestStreamBuilder>::value,
                "RequestStreamBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::experimental::RequestTagBuilder>::value,
                "RequestTagBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::experimental::RequestTagBuilder>::value,
                "RequestTagBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::experimental::RequestTagBuilder>::value,
                "RequestTagBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::experimental::RequestTagMultiBuilder>::value,
                "RequestTagMultiBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::experimental::RequestTagMultiBuilder>::value,
                "RequestTagMultiBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::experimental::RequestTagMultiBuilder>::value,
                "RequestTagMultiBuilder must remain move constructible");
}

TEST(RequestBuilderTraitsTest, BuildAndConversionRequireNonConstBuilders)
{
  assertBuildRequiresNonConstBuilder<ucxx::experimental::RequestAmBuilder, ucxx::RequestAm>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestAmBuilder, ucxx::RequestAm>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestAmBuilder, ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::experimental::RequestEndpointCloseBuilder,
                                     ucxx::RequestEndpointClose>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestEndpointCloseBuilder,
                                          ucxx::RequestEndpointClose>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestEndpointCloseBuilder,
                                          ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::experimental::RequestFlushBuilder, ucxx::RequestFlush>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestFlushBuilder,
                                          ucxx::RequestFlush>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestFlushBuilder, ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::experimental::RequestMemBuilder, ucxx::RequestMem>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestMemBuilder,
                                          ucxx::RequestMem>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestMemBuilder, ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::experimental::RequestStreamBuilder,
                                     ucxx::RequestStream>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestStreamBuilder,
                                          ucxx::RequestStream>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestStreamBuilder,
                                          ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::experimental::RequestTagBuilder, ucxx::RequestTag>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestTagBuilder,
                                          ucxx::RequestTag>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestTagBuilder, ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::experimental::RequestTagMultiBuilder,
                                     ucxx::RequestTagMulti>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestTagMultiBuilder,
                                          ucxx::RequestTagMulti>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RequestTagMultiBuilder,
                                          ucxx::Request>();
}

TEST(RequestBuilderSingleUseTest, BuildAttemptMarksBuilderBuilt)
{
  auto builder = ucxx::experimental::createRequestFlush(std::shared_ptr<ucxx::Component>{nullptr},
                                                        ucxx::data::Flush{});

  EXPECT_THROW(std::ignore = builder.build(), ucxx::Error);
  EXPECT_THROW(std::ignore = builder.build(), std::logic_error);
}

TEST(RequestBuilderSingleUseTest, ImplicitConversionAttemptMarksBuilderBuilt)
{
  auto builder = ucxx::experimental::createRequestFlush(std::shared_ptr<ucxx::Component>{nullptr},
                                                        ucxx::data::Flush{});

  EXPECT_THROW(std::ignore = static_cast<std::shared_ptr<ucxx::Request>>(builder), ucxx::Error);
  EXPECT_THROW(std::ignore = static_cast<std::shared_ptr<ucxx::Request>>(builder),
               std::logic_error);
}

TEST_F(RequestBuilderTest, EndpointFlushReturnsRequest)
{
  auto req = _ep->flush();
  static_assert(std::is_same<decltype(req), std::shared_ptr<ucxx::Request>>::value,
                "ep->flush() returns shared_ptr<Request>");
  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, WorkerFlushReturnsRequest)
{
  auto req = _worker->flush();
  static_assert(std::is_same<decltype(req), std::shared_ptr<ucxx::Request>>::value,
                "worker->flush() returns shared_ptr<Request>");
  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, EndpointWorkerTagSendRecvReturnRequests)
{
  std::vector<int> sendBuf{10, 20, 30};
  std::vector<int> recvBuf(3);
  auto tag     = ucxx::Tag{42};
  auto tagMask = ucxx::TagMaskFull;

  auto sendReq = _ep->tagSend(sendBuf.data(), sendBuf.size() * sizeof(int), tag);
  auto recvReq = _worker->tagRecv(recvBuf.data(), recvBuf.size() * sizeof(int), tag, tagMask);
  static_assert(std::is_same<decltype(sendReq), std::shared_ptr<ucxx::Request>>::value,
                "ep->tagSend() returns shared_ptr<Request>");
  static_assert(std::is_same<decltype(recvReq), std::shared_ptr<ucxx::Request>>::value,
                "worker->tagRecv() returns shared_ptr<Request>");

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();

  EXPECT_EQ(sendBuf, recvBuf);
}

TEST_F(RequestBuilderTest, EndpointTagSendCallback)
{
  std::vector<int> sendBuf{1, 2};
  std::vector<int> recvBuf(2);
  auto tag     = ucxx::Tag{99};
  auto tagMask = ucxx::TagMaskFull;

  bool callbackCalled                  = false;
  ucxx::RequestCallbackUserFunction cb = [&callbackCalled](ucs_status_t, std::shared_ptr<void>) {
    callbackCalled = true;
  };

  auto sendReq = _ep->tagSend(sendBuf.data(), sendBuf.size() * sizeof(int), tag, false, cb);
  auto recvReq = _worker->tagRecv(recvBuf.data(), recvBuf.size() * sizeof(int), tag, tagMask);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();

  EXPECT_EQ(sendBuf, recvBuf);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(RequestBuilderTest, EndpointTagSendAutoDeducesRequest)
{
  // Verify that `auto req = ep->tagSend(...)` preserves the legacy request type.
  std::vector<int> sendBuf{9, 8, 7};
  std::vector<int> recvBuf(3);
  auto tag     = ucxx::Tag{55};
  auto tagMask = ucxx::TagMaskFull;

  auto sendReq = _ep->tagSend(sendBuf.data(), sendBuf.size() * sizeof(int), tag);
  auto recvReq = _worker->tagRecv(recvBuf.data(), recvBuf.size() * sizeof(int), tag, tagMask);

  static_assert(std::is_same<decltype(sendReq), std::shared_ptr<ucxx::Request>>::value,
                "auto ep->tagSend() deduces shared_ptr<Request>");

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();

  EXPECT_EQ(sendBuf, recvBuf);
}

}  // namespace
