/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include <ucxx/utils/callback_notifier.h>

namespace {

TEST(CallbackNotifierTest, SetBeforeWait)
{
  ucxx::utils::CallbackNotifier notifier{};
  notifier.set();
  ASSERT_TRUE(notifier.wait());
}

TEST(CallbackNotifierTest, WaitTimeout)
{
  ucxx::utils::CallbackNotifier notifier{};
  size_t signalCount = 0;
  ASSERT_FALSE(
    notifier.wait(10000000 /* 10ms */, [&signalCount]() { ++signalCount; }, 1000000 /* 1ms */));
  ASSERT_EQ(signalCount, 10u);
}

TEST(CallbackNotifierTest, WaitStopsSignalingOnceSet)
{
  ucxx::utils::CallbackNotifier notifier{};
  notifier.set();

  size_t signalCount = 0;
  ASSERT_TRUE(
    notifier.wait(1000000000 /* 1s */, [&signalCount]() { ++signalCount; }, 1000000 /* 1ms */));
  // Once the flag is observed `wait()` must return immediately, without signaling.
  ASSERT_EQ(signalCount, 0u);
}

/**
 * Stress the lifetime contract of `CallbackNotifier`: the waiter owns the notifier and
 * destroys it as soon as `wait()` returns, while `set()` is called from another thread.
 * `set()` must not access the notifier after the waiter may observe the flag (previously
 * `notify_one()` was called after unlocking the mutex, a use-after-free). A short signal
 * interval makes the waiter frequently observe the flag on timed wake-ups rather than on
 * notification, widening the race window. The notifier is heap-allocated so that
 * ThreadSanitizer/AddressSanitizer builds reliably report invalid accesses.
 */
TEST(CallbackNotifierTest, DestroyImmediatelyAfterWait)
{
  constexpr size_t iterations = 20000;

  std::atomic<ucxx::utils::CallbackNotifier*> pending{nullptr};
  std::atomic<bool> stop{false};

  std::thread setter([&pending, &stop]() {
    while (!stop.load(std::memory_order_acquire)) {
      auto notifier = pending.exchange(nullptr, std::memory_order_acq_rel);
      if (notifier != nullptr)
        notifier->set();
      else
        std::this_thread::yield();
    }
  });

  for (size_t i = 0; i < iterations; ++i) {
    auto notifier = std::make_unique<ucxx::utils::CallbackNotifier>();
    pending.store(notifier.get(), std::memory_order_release);
    EXPECT_TRUE(notifier->wait(10000000000 /* 10s */, nullptr, 1000 /* 1us */));
    // `notifier` is destroyed here, possibly while `set()` is still returning.
  }

  stop.store(true, std::memory_order_release);
  setter.join();
}

}  // namespace
