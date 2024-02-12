/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <atomic>
#include <condition_variable>
#include <mutex>
namespace ucxx {

namespace utils {

/**
 * A thread-safe notification object.
 *
 * A thread-safe notification object which can signal release of some shared state while
 * a single thread blocks until the shared state is released.
 */
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
   * release of some shared state with `set()` while a single thread
   * blocks on `wait()` until the shared state is released.
   *
   * If libc is glibc and the version is older than 2.25, the
   * implementation uses a spinlock otherwise it uses a condition
   * variable.
   *
   * When C++-20 is the minimum supported version, it should use
   * atomic.wait + notify_one.
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
   * Set the flag to true and notify a single thread blocked by a call to `wait()`.
   * See also `std::condition_variable::notify_one`.
   */
  void set();

  /**
   * @brief Wait until `set()` has been called or period has elapsed.
   *
   * Wait until `set()` has been called, or period (in nanoseconds) has elapsed (only
   * applicable if using glibc 2.25 and higher).
   *
   * See also `std::condition_variable::wait`.
   *
   * @param[in] period  maximum period in nanoseconds to wait for or `0` to wait forever.
   *
   * @return  `true` if waiting finished or `false` if a timeout occurred.
   */
  bool wait(uint64_t period = 0);
};

}  // namespace utils
}  // namespace ucxx
