/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>

namespace ucxx {

class Future;
class Notifier;

namespace utils {

/*
 * @brief Check whether Python support is available.
 *
 * Check that binary were built with Python support and `libucxx_python.so` is in the
 * library path. The check is done by attempting to `dlopen` the library, returning whether
 * both conditions are met.
 *
 * @code{.cpp}
 * // worker is `std::shared_ptr<ucxx::Worker>`
 *
 * // Block until UCX's wakes up for an incoming event, then fully progresses the
 * // worker
 * worker->initBlockingProgressMode();
 * worker->progressWorkerEvent();
 *
 * // All events have been progressed.
 * @endcode
 *
 * @throws std::ios_base::failure if creating any of the file descriptors or setting their
 *                                statuses.
 * @returns whether Python support is available.
 */
bool isPythonAvailable();

/*
 * @brief Create a Python future if Python support is available.
 *
 * Create a Python future by loading the `ucxx::python::createPythonFuture` symbol at
 * runtime if `isPythonAvailable()` returns `true` and the symbol is found, otherwise
 * returns a `nullptr`.
 *
 * @returns the Python future or `nullptr`.
 */
std::shared_ptr<::ucxx::Future> createPythonFuture(std::shared_ptr<::ucxx::Notifier> notifier);

/*
 * @brief Create a Python notifier if Python support is available.
 *
 * Create a Python future by loading the `ucxx::python::createPythonNotifier` symbol at
 * runtime if `isPythonAvailable()` returns `true` and the symbol is found, otherwise
 * returns a `nullptr`.
 *
 * @returns the Python future or `nullptr`.
 */
std::shared_ptr<::ucxx::Notifier> createPythonNotifier();

}  // namespace utils

}  // namespace ucxx
