/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <gtest/gtest.h>

#include <ucxx/api.h>

namespace {
TEST(ConfigTest, HandleIsValid)
{
  ucxx::ConfigMap configMap{};
  ucxx::Config config{configMap};

  ASSERT_TRUE(config.getHandle() != nullptr);
}

// TEST(ConfigTest, ConfigMapDefault) {
//     ucxx::Config config{};

//     auto configMapOut = config.get();

//     // ASSERT_GT(configMapOut.size(), 1u);
//     // ASSERT_NE(configMapOut.find("TLS"), configMapOut.end());
//     // ASSERT_EQ(configMapOut["TLS"], "all");
//     for (const auto it : configMapOut) {
//         std::cout << it.first << ": " << it.second << std::endl;
//     }
// }

TEST(ConfigTest, ConfigMapTLS)
{
  ucxx::ConfigMap configMap{{"UCX_TLS", "tcp"}};
  ucxx::Config config{configMap};

  auto configMapOut = config.get();

  ASSERT_GT(configMapOut.size(), 1u);
  ASSERT_NE(configMapOut.find("TLS"), configMapOut.end());
  ASSERT_EQ(configMapOut["TLS"], "tcp");
  // auto configMapOut = config.get();
  // for (const auto it : configMapOut) {
  //     std::cout << it.first << ": " << it.second << std::endl;
  // }
}
}  // namespace

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
