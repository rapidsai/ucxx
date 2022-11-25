/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#if UCXX_ENABLE_PYTHON
#include <Python.h>

namespace ucxx {

namespace python {

PyObject* create_python_future();

PyObject* future_set_result(PyObject* future, PyObject* value);

PyObject* future_set_exception(PyObject* future, PyObject* exception, const char* message);

}  // namespace python

}  // namespace ucxx

#endif
