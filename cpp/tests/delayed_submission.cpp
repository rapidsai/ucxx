/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <stdexcept>

#include <gtest/gtest.h>

#include <ucxx/delayed_submission.h>

namespace {

TEST(GenericDelayedSubmissionCollectionTest, CancelPending)
{
  ucxx::GenericDelayedSubmissionCollection collection{"test"};
  size_t calls = 0;

  auto id = collection.schedule([&calls]() { ++calls; });
  ASSERT_TRUE(collection.cancel(id));

  collection.process();
  ASSERT_EQ(calls, 0u);
}

TEST(GenericDelayedSubmissionCollectionTest, CancelAlreadyProcessed)
{
  ucxx::GenericDelayedSubmissionCollection collection{"test"};
  size_t calls = 0;

  auto id = collection.schedule([&calls]() { ++calls; });
  collection.process();
  ASSERT_EQ(calls, 1u);

  // The callback already completed, canceling must report it was not pending anymore.
  ASSERT_FALSE(collection.cancel(id));

  // A subsequent callback must not be affected.
  auto id2 = collection.schedule([&calls]() { ++calls; });
  collection.process();
  ASSERT_EQ(calls, 2u);
  ASSERT_FALSE(collection.cancel(id2));
}

TEST(GenericDelayedSubmissionCollectionTest, CancelWhileProcessingThrows)
{
  ucxx::GenericDelayedSubmissionCollection collection{"test"};
  bool threw = false;
  ucxx::ItemIdType id{};

  id = collection.schedule([&collection, &id, &threw]() {
    try {
      collection.cancel(id);
    } catch (const std::runtime_error&) {
      threw = true;
    }
  });
  collection.process();
  ASSERT_TRUE(threw);
}

}  // namespace
