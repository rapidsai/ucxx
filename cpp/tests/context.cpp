/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdlib>
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

INSTANTIATE_TEST_SUITE_P(TLS, ContextTestCustomConfig, testing::ValuesIn(TlsConfig));

}  // namespace
