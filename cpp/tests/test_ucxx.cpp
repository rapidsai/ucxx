/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <gtest/gtest.h>

// TODO: Remove after CMake
#include "buffer.cpp"
#include "config.cpp"
#include "context.cpp"
#include "endpoint.cpp"
#include "header.cpp"
#include "listener.cpp"
#include "request.cpp"
#include "worker.cpp"

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
