/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <atomic>
#include <condition_variable>
#include <mutex>
namespace ucxx {

namespace utils {

class CallbackNotifier {
 private:
  std::atomic_bool _spinlock{};                  //< spinlock if we're not using condition variables
  bool _flag{};                                  //< flag storing state
  std::mutex _mutex{};                           //< lock to guard accesses
  std::condition_variable _conditionVariable{};  //< notification condition var
  bool _use_spinlock{};                          //< should we use the spinlock?
  bool use_spinlock();

 public:
  /**
   * @brief Construct a thread-safe notification object with given initial value.
   *
   * Construct a thread-safe notification object which can signal
   * release of some shared state with `set()` while other threads
   * block on `wait()` until the shared state is released.
   *
   * If libc is glibc and the version is older than 2.25, this uses a
   * spinlock otherwise it uses a condition variable,
   *
   * @param[in] init  The initial flag value
   */
  CallbackNotifier() : _flag{false}, _spinlock{false}, _use_spinlock{use_spinlock()} {};

  ~CallbackNotifier() = default;

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
  void set();

  /**
   * @brief Wait until set has been called
   *
   * See also `std::condition_variable::wait`
   */
  void wait();
};

}  // namespace utils
}  // namespace ucxx
