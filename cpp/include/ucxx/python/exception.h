/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#if UCXX_ENABLE_PYTHON
#include <Python.h>

#include <ucp/api/ucp.h>

namespace ucxx {

namespace python {

extern PyObject* UCXXBaseException;
extern PyObject* UCXXError;
extern PyObject* UCXXCanceledError;
extern PyObject* UCXXCloseError;
extern PyObject* UCXXConfigError;
extern PyObject* UCXXConnectionResetError;
extern PyObject* UCXXMessageTruncatedError;

/**
 * @brief Create Python exceptions.
 *
 * Create UCXX-specific Python exceptions such that they are visible both from C/C++ and
 * Python.
 */
void create_exceptions();

/**
 * @brief Raise a C++ exception in Python.
 *
 * Raise a C++ exception in Python. When a C++ exception occurs, Python needs to be able
 * to be informed of such event and be able to raise a Python exception from it. This
 * function raises both general C++ exceptions, as well as UCXX-specific exceptions.
 *
 * To use this, C++ methods and functions that are exposed to Python via Cython must have
 * a `except +raise_py_error` as a declaration suffix.
 */
void raise_py_error();

/**
 * @brief Get a Python exception from UCS status.
 *
 * Given a UCS status, get a matching Python exception object.
 *
 * @param[in] status UCS status from which to get exception.
 */
PyObject* get_python_exception_from_ucs_status(ucs_status_t status);

}  // namespace python

}  // namespace ucxx

#endif
