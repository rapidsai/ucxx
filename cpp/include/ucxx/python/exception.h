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

void create_exceptions();

void raise_py_error();

PyObject* get_python_exception_from_ucs_status(ucs_status_t status);

}  // namespace python

}  // namespace ucxx

#endif
