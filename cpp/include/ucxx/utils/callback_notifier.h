/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#define UCXX_USE_SPINLOCK 1
#if (UCXX_USE_SPINLOCK == 1)
#include <atomic>
#else
#include <condition_variable>
#include <mutex>
#endif
namespace ucxx {

namespace utils {

class CallbackNotifier {
 private:
#if (UCXX_USE_SPINLOCK == 1)
  std::atomic_bool _flag{};  //< flag storing state
#else
  bool _flag{};                                  //< flag storing state
  std::mutex _mutex{};                           //< lock to guard accesses
  std::condition_variable _conditionVariable{};  //< notification condition var
#endif
 public:
  /**
   * @brief Construct a thread-safe notification object with given initial value.
   *
   * Construct a thread-safe notification object which can signal
   * release of some shared state with `set()` while other threads
   * block on `wait()` until the shared state is released.
   *
   * If `UCXX_USE_SPINLOCK` is 0, this uses a condition variable,
     otherwise it uses an atomic spinlock.
   *
   * @param[in] init  The initial flag value
   */
  CallbackNotifier() : _flag{false} {}

  ~CallbackNotifier() {}

  CallbackNotifier(const CallbackNotifier&)            = delete;
  CallbackNotifier& operator=(CallbackNotifier const&) = delete;
  CallbackNotifier(CallbackNotifier&& o)               = delete;
  CallbackNotifier& operator=(CallbackNotifier&& o)    = delete;

  /**
   * @brief Notify waiting threads that we are done and they can proceed
   *
   * Set the flag to true and notify others threads blocked by a call to `wait()`.
   * See also `std::condition_variable::notify_all`.
   */
  void set()
  {
#if (UCXX_USE_SPINLOCK == 1)
    _flag.store(true, std::memory_order_release);
#else
    {
      std::lock_guard lock(_mutex);
      _flag = true;
    }
    _conditionVariable.notify_all();
#endif
  }

  /**
   * @brief Wait until set has been called
   *
   * See also `std::condition_variable::wait`
   */
  void wait()
  {
#if (UCXX_USE_SPINLOCK == 1)
    while (!_flag.load(std::memory_order_acquire)) {}
#else
    std::unique_lock lock(_mutex);
    _conditionVariable.wait(lock, [this]() { return _flag; });
#endif
  }
};

}  // namespace utils
//
}  // namespace ucxx
