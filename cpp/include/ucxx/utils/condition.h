/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>

namespace ucxx {

namespace utils {

/**
 * @brief Run user-defined function and notify condition.
 *
 * Run a user-defined function `func` within a guarded scope defined by `m`, upon completion
 * notify `cond`. Generally useful to run functions within the scope of
 * `ucxx::Worker::registerGenericPre` and `ucxx::worker::registerGenericPost`, such that the
 * caller can check for completion of the function via `conditionGetter`.
 *
 * @param[in] m     mutex to lock.
 * @param[in] cond  conditional variable to notify when `func` completes.
 * @param[in] func  user-defined function to execute.
 */
template <typename Func>
void conditionSetter(std::shared_ptr<std::mutex> m,
                     std::shared_ptr<std::condition_variable> cond,
                     Func func)
{
  {
    std::lock_guard<std::mutex> lock(*m);
    func();
  }
  cond->notify_one();
}

/**
 * @brief Run user-defined function to check when `conditionSetter` completes.
 *
 * Run a user-defined function `func` within a guarded scope defined by `m`, checking that
 * `result` evaluates to `true` when `cond` is notified, which is executed upon completion
 * completion of `conditionSetter`. Generally useful to check `conditionSetter` completed
 * running in a separate thread, such as in the scope of to run functions within the scope of
 * `ucxx::Worker::registerGenericPre` or `ucxx::worker::registerGenericPost`.
 *
 * @param[in] m       mutex to lock.
 * @param[in] cond    conditional variable to notify when `func` completes.
 * @param[in] result  the result to check.
 * @param[in] func    user-defined function to execute.
 */
template <typename Result, typename Func>
void conditionGetter(std::shared_ptr<std::mutex> m,
                     std::shared_ptr<std::condition_variable> cond,
                     std::shared_ptr<Result> result,
                     Func func)
{
  std::unique_lock<std::mutex> lock(*m);
  cond->wait(lock, [result, func] { return func(); });
}

}  // namespace utils

}  // namespace ucxx
