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
  std::atomic_bool _flag{};                      //< flag storing state
  std::mutex _mutex{};                           //< lock to guard accesses
  std::condition_variable _conditionVariable{};  //< notification condition var

 public:
  /**
   * @brief Construct a thread-safe notification object
   *
   * Construct a thread-safe notification object which can signal
   * release of some shared state with `set()` while other threads
   * block on `wait()` until the shared state is released.
   *
   * If libc is glibc and the version is older than 2.25, the
   * implementation uses a spinlock otherwise it uses a condition
   * variable.
   *
   * When C++-20 is the minimum supported version, it should use
   * atomic.wait + notify_all.
   */
  CallbackNotifier() : _flag{false} {};

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
   * @brief Wait until `set()` has been called
   *
   * See also `std::condition_variable::wait`.
   */
  void wait();
};

}  // namespace utils
}  // namespace ucxx
