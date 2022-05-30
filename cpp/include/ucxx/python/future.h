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

static int intern_strings(void);

static int init_ucxx_python();

static PyObject* get_asyncio_future_object();

static PyObject* create_python_future();

static PyCFunction get_future_method(const char* method_name);

static PyObject* future_set_result(PyObject* future, PyObject* value);

static PyObject* future_set_exception(PyObject* future, PyObject* exception, const char* message);

}  // namespace python

}  // namespace ucxx

#endif
