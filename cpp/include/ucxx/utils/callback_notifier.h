/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <condition_variable>
#include <memory>
#include <mutex>

namespace ucxx {

namespace utils {

template <typename Flag>
class CallbackNotifier {
 private:
  std::shared_ptr<std::atomic<Flag>> _flag{};    //< flag to spin on
  std::function<void()> _function{};             //< function to run before setting flag
  std::mutex _mutex{};                           //< lock to guard accesses
  std::condition_variable _conditionVariable{};  //< notification condition var

 public:
  /**
   * @brief Construct a thread-safe notification object with given initial value
   *
   * @param[in] init   The initial flag value
   */
  CallbackNotifier(std::shared_ptr<std::atomic<Flag>> flag,
                   std::function<void()> function = nullptr)
    : _flag{flag}, _function{function} {};

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
      if (_function) _function();
      _flag->store(flag);
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
  std::shared_ptr<std::atomic<Flag>> wait(Compare compare)
  {
    std::unique_lock lock(_mutex);
    _conditionVariable.wait(lock, [this, &compare]() { return compare(_flag); });
    return _flag;
  }
};

}  // namespace utils
//
}  // namespace ucxx
