/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ucxx/api.h>

namespace {

static std::vector<std::string> TlsConfig{"^tcp", "^tcp,sm", "tcp", "tcp,sm", "all"};

class ContextTestCustomConfig : public testing::TestWithParam<std::string> {};

TEST(ContextTest, HandleIsValid)
{
  auto context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);

  ASSERT_TRUE(context->getHandle() != nullptr);
}

TEST(ContextTest, DefaultConfigsAndFlags)
{
  static constexpr auto featureFlags = ucxx::Context::defaultFeatureFlags;
  auto context                       = ucxx::createContext({}, featureFlags);
  auto configMapOut                  = context->getConfig();
  ASSERT_GT(configMapOut.size(), 1u);
  ASSERT_NE(configMapOut.find("TLS"), configMapOut.end());
  if (const char* envTls = std::getenv("UCX_TLS"))
    ASSERT_EQ(configMapOut["TLS"], envTls);
  else
    ASSERT_EQ(configMapOut["TLS"], "all");

  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
}

TEST_P(ContextTestCustomConfig, TLS)
{
  auto tls                           = GetParam();
  static constexpr auto featureFlags = ucxx::Context::defaultFeatureFlags;
  auto context                       = ucxx::createContext({{"TLS", tls}}, featureFlags);

  auto configMapOut = context->getConfig();
  ASSERT_GT(configMapOut.size(), 1u);
  ASSERT_NE(configMapOut.find("TLS"), configMapOut.end());
  ASSERT_EQ(configMapOut["TLS"], tls);

  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
}

TEST(ContextTest, CustomFlags)
{
  uint64_t featureFlags = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;
  auto context          = ucxx::createContext({}, featureFlags);

  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
}

TEST(ContextTest, Info)
{
  auto context = ucxx::createContext({{"UCX_TLS", "tcp"}}, ucxx::Context::defaultFeatureFlags);

  ASSERT_GT(context->getInfo().size(), 0u);
}

TEST(ContextTest, CreateWorker)
{
  auto context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);

  auto worker1 = ucxx::createWorker(context, false, false);
  ASSERT_TRUE(worker1 != nullptr);

  auto worker2 = context->createWorker();
  ASSERT_TRUE(worker2 != nullptr);
}

// Builder Pattern Tests
TEST(ContextBuilderTest, BasicBuilderWithAuto)
{
  auto context = ucxx::createContext().build();
  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), ucxx::Context::defaultFeatureFlags);
}

TEST(ContextBuilderTest, BuilderWithFeatureFlags)
{
  uint64_t featureFlags = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;
  auto context          = ucxx::createContext().featureFlags(featureFlags).build();

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
}

TEST(ContextBuilderTest, BuilderWithConfigMap)
{
  auto context = ucxx::createContext().configMap({{"TLS", "tcp"}}).build();

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  auto config = context->getConfig();
  ASSERT_EQ(config["TLS"], "tcp");
}

TEST(ContextBuilderTest, BuilderMethodChainingConfigFirst)
{
  uint64_t featureFlags = UCP_FEATURE_RMA | UCP_FEATURE_STREAM;
  auto context =
    ucxx::createContext().configMap({{"TLS", "all"}}).featureFlags(featureFlags).build();

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
  auto config = context->getConfig();
  ASSERT_EQ(config["TLS"], "all");
}

TEST(ContextBuilderTest, BuilderMethodChainingFlagsFirst)
{
  uint64_t featureFlags = UCP_FEATURE_AM | UCP_FEATURE_RMA;
  auto context =
    ucxx::createContext().featureFlags(featureFlags).configMap({{"TLS", "tcp"}}).build();

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
  auto config = context->getConfig();
  ASSERT_EQ(config["TLS"], "tcp");
}

TEST(ContextBuilderTest, BuilderExplicitTypeSpecification)
{
  uint64_t featureFlags                  = UCP_FEATURE_TAG;
  std::shared_ptr<ucxx::Context> context = ucxx::createContext().featureFlags(featureFlags);

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
}

TEST(ContextBuilderTest, BuilderAutoTypes)
{
  auto builder1 = ucxx::createContext();
  static_assert(std::is_same<decltype(builder1), ucxx::ContextBuilder>::value,
                "auto without .build() is ContextBuilder");

  auto builder2 = ucxx::createContext().featureFlags(UCP_FEATURE_TAG);
  static_assert(std::is_same<decltype(builder2), ucxx::ContextBuilder>::value,
                "auto with only .featureFlags() is ContextBuilder");

  auto context1 = builder1.build();
  static_assert(std::is_same<decltype(context1), std::shared_ptr<ucxx::Context>>::value,
                "Calling .build() on builder returns shared_ptr<Context>");

  std::shared_ptr<ucxx::Context> context2 = builder2;
  static_assert(std::is_same<decltype(context2), std::shared_ptr<ucxx::Context>>::value,
                "Implicit conversion with explicit type works");

  auto context3 = ucxx::createContext().build();
  static_assert(std::is_same<decltype(context1), std::shared_ptr<ucxx::Context>>::value,
                "auto with .build() must be shared_ptr<Context>");

  auto context4 = ucxx::createContext().featureFlags(UCP_FEATURE_TAG).build();
  static_assert(std::is_same<decltype(context2), std::shared_ptr<ucxx::Context>>::value,
                "auto with .featureFlags().build() must be shared_ptr<Context>");

  auto context5 = ucxx::createContext().configMap({{"TLS", "tcp"}}).build();
  static_assert(std::is_same<decltype(context3), std::shared_ptr<ucxx::Context>>::value,
                "auto with .configMap().build() must be shared_ptr<Context>");

  auto context6 =
    ucxx::createContext().configMap({{"TLS", "all"}}).featureFlags(UCP_FEATURE_RMA).build();
  static_assert(std::is_same<decltype(context4), std::shared_ptr<ucxx::Context>>::value,
                "auto with full chain and .build() must be shared_ptr<Context>");

  ASSERT_TRUE(context1 != nullptr);
  ASSERT_TRUE(context2 != nullptr);
  ASSERT_TRUE(context3 != nullptr);
  ASSERT_TRUE(context4 != nullptr);
  ASSERT_TRUE(context5 != nullptr);
  ASSERT_TRUE(context6 != nullptr);
}

TEST(ContextBuilderTest, BuilderImplicitConversion)
{
  std::shared_ptr<ucxx::Context> context = ucxx::createContext().featureFlags(UCP_FEATURE_RMA);

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), UCP_FEATURE_RMA);
}

TEST(ContextBuilderTest, BuilderSingleConstructionPerSet)
{
  auto builder = ucxx::createContext().featureFlags(UCP_FEATURE_TAG);
  auto context = builder.build();

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), UCP_FEATURE_TAG);

  // Same builder can be used multiple times, but creates separate contexts
  auto context2 = builder.build();
  ASSERT_TRUE(context2 != nullptr);
  ASSERT_NE(context->getHandle(), context2->getHandle());  // Different contexts
}

TEST(ContextBuilderTest, BuilderDifferentInstances)
{
  auto context1 = ucxx::createContext().featureFlags(UCP_FEATURE_TAG).build();
  auto context2 = ucxx::createContext().featureFlags(UCP_FEATURE_TAG).build();

  // Different builder chains should create different contexts
  ASSERT_TRUE(context1 != nullptr);
  ASSERT_TRUE(context2 != nullptr);
  ASSERT_NE(context1->getHandle(), context2->getHandle());
}

TEST(ContextBuilderTest, BuilderBackwardCompatibility)
{
  uint64_t featureFlags = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;

  // Old API should still work
  auto context1 = ucxx::createContext({{"TLS", "tcp"}}, featureFlags);
  ASSERT_TRUE(context1 != nullptr);
  ASSERT_TRUE(context1->getHandle() != nullptr);
  ASSERT_EQ(context1->getFeatureFlags(), featureFlags);
  auto config1 = context1->getConfig();
  ASSERT_EQ(config1["TLS"], "tcp");

  // New API should produce equivalent result
  auto context2 =
    ucxx::createContext().configMap({{"TLS", "tcp"}}).featureFlags(featureFlags).build();
  ASSERT_TRUE(context2 != nullptr);
  ASSERT_TRUE(context2->getHandle() != nullptr);
  ASSERT_EQ(context2->getFeatureFlags(), featureFlags);
  auto config2 = context2->getConfig();
  ASSERT_EQ(config2["TLS"], "tcp");

  // Both should have same configuration
  ASSERT_EQ(context1->getFeatureFlags(), context2->getFeatureFlags());
  ASSERT_EQ(config1["TLS"], config2["TLS"]);
}

TEST(ContextBuilderTest, BuilderContextIsValid)
{
  auto context = ucxx::createContext().featureFlags(ucxx::Context::defaultFeatureFlags).build();

  auto worker = context->createWorker();
  ASSERT_TRUE(worker != nullptr);
}

TEST_P(ContextTestCustomConfig, BuilderTLS)
{
  auto tls                           = GetParam();
  static constexpr auto featureFlags = ucxx::Context::defaultFeatureFlags;
  auto context = ucxx::createContext().configMap({{"TLS", tls}}).featureFlags(featureFlags).build();

  auto configMapOut = context->getConfig();
  ASSERT_GT(configMapOut.size(), 1u);
  ASSERT_NE(configMapOut.find("TLS"), configMapOut.end());
  ASSERT_EQ(configMapOut["TLS"], tls);

  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
}

INSTANTIATE_TEST_SUITE_P(TLS, ContextTestCustomConfig, testing::ValuesIn(TlsConfig));

}  // namespace
