/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <ucxx/api.h>
#include <ucxx/experimental/address_builder.h>
#include <ucxx/experimental/endpoint_builder.h>
#include <ucxx/experimental/listener_builder.h>
#include <ucxx/experimental/memory_handle_builder.h>
#include <ucxx/experimental/remote_key_builder.h>
#include <ucxx/experimental/tag_probe_builder.h>

namespace {

static void listenerCallback(ucp_conn_request_h, void*) {}

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

class ComponentBuilderTest : public ::testing::Test {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::experimental::createContext(ucxx::Context::defaultFeatureFlags).build()};
  std::shared_ptr<ucxx::Worker> _worker{ucxx::experimental::createWorker(_context).build()};
};

TEST_F(ComponentBuilderTest, AddressBuilderFromWorkerAndString)
{
  auto builder = ucxx::experimental::createAddressFromWorker(_worker);
  static_assert(std::is_same<decltype(builder), ucxx::experimental::AddressBuilder>::value,
                "createAddressFromWorker returns AddressBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::experimental::AddressBuilder, ucxx::Address>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::AddressBuilder, ucxx::Address>();

  auto address = builder.build();
  static_assert(std::is_same<decltype(address), std::shared_ptr<ucxx::Address>>::value,
                "AddressBuilder::build returns shared_ptr<Address>");
  ASSERT_TRUE(address != nullptr);
  ASSERT_TRUE(address->getHandle() != nullptr);
  ASSERT_GT(address->getLength(), 0u);

  std::string addressString{address->getStringView()};
  std::shared_ptr<ucxx::Address> copiedAddress =
    ucxx::experimental::createAddressFromString(addressString);
  ASSERT_TRUE(copiedAddress != nullptr);
  ASSERT_TRUE(copiedAddress->getHandle() != nullptr);
  ASSERT_EQ(copiedAddress->getStringView(), addressString);
}

TEST_F(ComponentBuilderTest, EndpointBuilderFromWorkerAddress)
{
  auto builder = ucxx::experimental::createEndpointFromWorkerAddress(_worker, _worker->getAddress())
                   .endpointErrorHandling(false);
  static_assert(std::is_same<decltype(builder), ucxx::experimental::EndpointBuilder>::value,
                "createEndpointFromWorkerAddress returns EndpointBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::experimental::EndpointBuilder, ucxx::Endpoint>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::EndpointBuilder, ucxx::Endpoint>();

  auto endpoint = builder.build();
  static_assert(std::is_same<decltype(endpoint), std::shared_ptr<ucxx::Endpoint>>::value,
                "EndpointBuilder::build returns shared_ptr<Endpoint>");
  ASSERT_TRUE(endpoint != nullptr);
  ASSERT_TRUE(endpoint->getHandle() != nullptr);
  ASSERT_TRUE(endpoint->isAlive());
}

TEST_F(ComponentBuilderTest, ListenerBuilder)
{
  auto builder = ucxx::experimental::createListener(_worker, 0, listenerCallback, nullptr);
  static_assert(std::is_same<decltype(builder), ucxx::experimental::ListenerBuilder>::value,
                "createListener returns ListenerBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::experimental::ListenerBuilder, ucxx::Listener>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::ListenerBuilder, ucxx::Listener>();

  auto listener = builder.build();
  static_assert(std::is_same<decltype(listener), std::shared_ptr<ucxx::Listener>>::value,
                "ListenerBuilder::build returns shared_ptr<Listener>");
  ASSERT_TRUE(listener != nullptr);
  ASSERT_TRUE(listener->getHandle() != nullptr);
  ASSERT_GT(listener->getPort(), 0u);
}

TEST_F(ComponentBuilderTest, MemoryHandleBuilder)
{
  std::vector<char> buffer(128);

  auto builder = ucxx::experimental::createMemoryHandle(_context, buffer.size())
                   .buffer(buffer.data())
                   .memoryType(UCS_MEMORY_TYPE_HOST);
  static_assert(std::is_same<decltype(builder), ucxx::experimental::MemoryHandleBuilder>::value,
                "createMemoryHandle returns MemoryHandleBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::experimental::MemoryHandleBuilder, ucxx::MemoryHandle>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::MemoryHandleBuilder,
                                          ucxx::MemoryHandle>();

  auto memoryHandle = builder.build();
  static_assert(std::is_same<decltype(memoryHandle), std::shared_ptr<ucxx::MemoryHandle>>::value,
                "MemoryHandleBuilder::build returns shared_ptr<MemoryHandle>");
  ASSERT_TRUE(memoryHandle != nullptr);
  ASSERT_TRUE(memoryHandle->getHandle() != nullptr);
  ASSERT_EQ(memoryHandle->getSize(), buffer.size());
  ASSERT_EQ(memoryHandle->getMemoryType(), UCS_MEMORY_TYPE_HOST);
}

TEST_F(ComponentBuilderTest, RemoteKeyBuilderFromMemoryHandleAndSerialized)
{
  auto memoryHandle = ucxx::experimental::createMemoryHandle(_context, 128).build();

  auto builder = ucxx::experimental::createRemoteKeyFromMemoryHandle(memoryHandle);
  static_assert(std::is_same<decltype(builder), ucxx::experimental::RemoteKeyBuilder>::value,
                "createRemoteKeyFromMemoryHandle returns RemoteKeyBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::experimental::RemoteKeyBuilder, ucxx::RemoteKey>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::RemoteKeyBuilder, ucxx::RemoteKey>();

  auto localRemoteKey = builder.build();
  static_assert(std::is_same<decltype(localRemoteKey), std::shared_ptr<ucxx::RemoteKey>>::value,
                "RemoteKeyBuilder::build returns shared_ptr<RemoteKey>");
  ASSERT_TRUE(localRemoteKey != nullptr);
  ASSERT_EQ(localRemoteKey->getSize(), memoryHandle->getSize());

  auto endpoint =
    ucxx::experimental::createEndpointFromWorkerAddress(_worker, _worker->getAddress()).build();
  std::shared_ptr<ucxx::RemoteKey> unpackedRemoteKey =
    ucxx::experimental::createRemoteKeyFromSerialized(endpoint, localRemoteKey->serialize());
  ASSERT_TRUE(unpackedRemoteKey != nullptr);
  ASSERT_TRUE(unpackedRemoteKey->getHandle() != nullptr);
  ASSERT_EQ(unpackedRemoteKey->getSize(), localRemoteKey->getSize());
}

TEST(ComponentBuilderStandaloneTest, TagProbeInfoBuilder)
{
  auto builder = ucxx::experimental::createTagProbeInfo();
  static_assert(std::is_same<decltype(builder), ucxx::experimental::TagProbeInfoBuilder>::value,
                "createTagProbeInfo returns TagProbeInfoBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::experimental::TagProbeInfoBuilder, ucxx::TagProbeInfo>();
  assertConversionRequiresNonConstBuilder<ucxx::experimental::TagProbeInfoBuilder,
                                          ucxx::TagProbeInfo>();

  auto unmatched = builder.build();
  static_assert(std::is_same<decltype(unmatched), std::shared_ptr<ucxx::TagProbeInfo>>::value,
                "TagProbeInfoBuilder::build returns shared_ptr<TagProbeInfo>");
  ASSERT_TRUE(unmatched != nullptr);
  ASSERT_FALSE(unmatched->isMatched());

  ucp_tag_recv_info_t info{};
  info.sender_tag = 7;
  info.length     = 64;

  std::shared_ptr<ucxx::TagProbeInfo> matched =
    ucxx::experimental::createTagProbeInfo(info, nullptr);
  ASSERT_TRUE(matched != nullptr);
  ASSERT_TRUE(matched->isMatched());
  ASSERT_EQ(matched->getInfo().senderTag, ucxx::Tag{7});
  ASSERT_EQ(matched->getInfo().length, 64u);
}

}  // namespace
