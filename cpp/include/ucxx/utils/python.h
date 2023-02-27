/**
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
 * Check that binary was built with Python support and `libucxx_python.so` is in the
 * library path. The check is done by attempting to `dlopen` the library, returning whether
 * both conditions are met.
 *
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
 * Create a Python notifier by loading the `ucxx::python::createPythonNotifier` symbol at
 * runtime if `isPythonAvailable()` returns `true` and the symbol is found, otherwise
 * returns a `nullptr`.
 *
 * @returns the Python notifier or `nullptr`.
 */
std::shared_ptr<::ucxx::Notifier> createPythonNotifier();

}  // namespace utils

}  // namespace ucxx
