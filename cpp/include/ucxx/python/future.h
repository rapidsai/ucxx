/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <Python.h>

namespace ucxx {

namespace python {

/**
 * @brief Create a Python asyncio future.
 *
 * Create Python asyncio future, effectively equal to calling `asyncio.Future()` directly
 * in Python.
 *
 * Note that this call will take the Python GIL and requires that the current thread have
 * an asynchronous event loop set.
 *
 * @returns The Python asyncio future object.
 */
PyObject* create_python_future();

/**
 * @brief Set the result of a Python future.
 *
 * Set the result of a Python future.
 *
 * Note that this call will take the Python GIL and requires that the current thread have
 * the same asynchronous event loop set as the thread that owns the future.
 *
 * @param[in] future  Python object containing the `_asyncio.Future` object.
 * @param[in] value   Python object containing an arbitrary value to set the future result
 *                    to.
 *
 * @returns The result of the call to `_asyncio.Future.set_result()`.
 */
PyObject* future_set_result(PyObject* future, PyObject* value);

/**
 * @brief Set the exception of a Python future.
 *
 * Set the exception of a Python future.
 *
 * Note that this call will take the Python GIL and requires that the current thread have
 * the same asynchronous event loop set as the thread that owns the future.
 *
 * @param[in] future    Python object containing the `_asyncio.Future` object.
 * @param[in] exception a Python exception derived of the `Exception` class.
 * @param[in] message   human-readable error message for the exception.
 *
 * @returns The result of the call to `_asyncio.Future.set_result()`.
 */
PyObject* future_set_exception(PyObject* future, PyObject* exception, const char* message);

}  // namespace python

}  // namespace ucxx
