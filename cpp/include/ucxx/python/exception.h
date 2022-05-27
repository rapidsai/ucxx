/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#if UCXX_ENABLE_PYTHON
#include <Python.h>

#include <ucp/api/ucp.h>

extern "C" {
extern PyObject* ucxx_error;
extern PyObject* ucxx_canceled_error;
extern PyObject* ucxx_config_error;
extern PyObject* ucxx_connection_reset_error;
}

namespace ucxx {

void raise_py_error();

PyObject* get_python_exception_from_ucs_status(ucs_status_t status);

}  // namespace ucxx

#endif
