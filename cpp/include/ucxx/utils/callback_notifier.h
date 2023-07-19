/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>

namespace ucxx {

namespace utils {

template <typename Flag>
class CallbackNotifier {
 private:
  Flag _flag{};
  std::mutex _mutex{};                           //< lock to guard accesses
  std::condition_variable _conditionVariable{};  //< notification condition var

 public:
  /**
   * @brief Construct a thread-safe notification object with given initial value
   *
   * @param[in] init   The initial flag value
   */
  explicit CallbackNotifier(Flag flag) : _flag{flag} {}

  ~CallbackNotifier() {}

  CallbackNotifier(const CallbackNotifier&) = delete;
  CallbackNotifier& operator=(CallbackNotifier const&) = delete;
  CallbackNotifier(CallbackNotifier&& o)               = delete;
  CallbackNotifier& operator=(CallbackNotifier&& o) = delete;

  /**
   * @brief Store a new flag value and notify a waiting thread
   *
   * @param[in] flag   New flag value
   */
  void store(Flag flag)
  {
    {
      std::lock_guard lock(_mutex);
      _flag = flag;
    }
    _conditionVariable.notify_one();
  }

  /**
   * @brief Wait while predicate is not true for the flag value to change
   *
   * @param[in] p   Function of type T -> bool called with the flag
   *                value. This function loops until the predicate is
   *                satisfied. See also std::condition_variable::wait
   * @param[out]    The new flag value
   */
  template <typename Compare>
  Flag wait(Compare compare)
  {
    std::unique_lock lock(_mutex);
    _conditionVariable.wait(lock, [this, &compare]() { return compare(_flag); });
    return std::move(_flag);
  }
};

}  // namespace utils
//
}  // namespace ucxx
