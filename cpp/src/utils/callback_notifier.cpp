/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "ucxx/log.h"
#include <features.h>
#include <iostream>
#include <ucxx/utils/callback_notifier.h>
#ifdef __GLIBC__
#include <gnu/libc-version.h>
#include <string>
#endif

namespace ucxx {
namespace utils {

#ifdef __GLIBC__
bool CallbackNotifier::use_spinlock()
{
  static bool use_set = false;
  static bool use     = false;
  if (use_set) return use;
  use_set           = true;
  auto libc_version = std::string_view{gnu_get_libc_version()};
  auto dot          = libc_version.find(".");
  if (dot == std::string::npos) {
    use = false;
  } else {
    int glibc_major = std::stoi(libc_version.substr(0, dot).data());
    int glibc_minor = std::stoi(libc_version.substr(dot + 1).data());
    use             = glibc_major < 2 || (glibc_major == 2 && glibc_minor < 25);
    ucxx_trace("glibc version %s detected, spinlock use is %d", libc_version.data(), use);
  }
  return use;
}
#else
bool CallbackNotifier::use_spinlock() { return false; }
#endif

void CallbackNotifier::set()
{
  if (use_spinlock()) {
    _spinlock.store(true, std::memory_order_release);
  } else {
    {
      std::lock_guard lock(_mutex);
      _flag = true;
    }
    _conditionVariable.notify_all();
  }
}
void CallbackNotifier::wait()
{
  if (use_spinlock()) {
    while (!_spinlock.load(std::memory_order_acquire)) {}
  } else {
    std::unique_lock lock(_mutex);
    _conditionVariable.wait(lock, [this]() { return _flag; });
  }
}

}  // namespace utils
}  // namespace ucxx
