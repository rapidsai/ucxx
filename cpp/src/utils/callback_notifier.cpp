/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <atomic>
#include <features.h>

#include <ucxx/log.h>
#include <ucxx/utils/callback_notifier.h>

namespace ucxx {
namespace utils {

void CallbackNotifier::set()
{
  {
    std::lock_guard lock(_mutex);
    // This can be relaxed because the mutex is providing
    // ordering.
    _flag.store(true, std::memory_order_relaxed);
  }
  _conditionVariable.notify_one();
}

bool CallbackNotifier::wait(uint64_t period)
{
  std::unique_lock lock(_mutex);
  // Likewise here, the mutex provides ordering.
  if (period > 0) {
    return _conditionVariable.wait_for(lock,
                                       std::chrono::duration<uint64_t, std::nano>(period),
                                       [this]() { return _flag.load(std::memory_order_relaxed); });
  } else {
    _conditionVariable.wait(lock, [this]() { return _flag.load(std::memory_order_relaxed); });
    return true;
  }
}

}  // namespace utils
}  // namespace ucxx
