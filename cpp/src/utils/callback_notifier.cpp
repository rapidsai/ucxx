/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <mutex>
#include <stdexcept>

#include <ucxx/utils/callback_notifier.h>

namespace ucxx {
namespace utils {

void CallbackNotifier::set()
{
  {
    // We're not allowed to change this value if we can't grab a lock
    std::lock_guard lock(_mutex);
    if (_flag) { throw std::runtime_error("Foo!"); }
    _flag = true;
  }
  _conditionVariable.notify_one();
}

void CallbackNotifier::wait()
{
  std::unique_lock lock(_mutex);
  _conditionVariable.wait(lock, [this]() { return _flag; });
  _flag = false;
}

}  // namespace utils
}  // namespace ucxx
