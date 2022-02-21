/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <exception>

#include "Python.h"

#include <ucxx/exception.h>

extern "C"
{
    extern PyObject *ucxx_error;
    extern PyObject *ucxx_config_error;
}

namespace ucxx
{

void raise_py_error()
{
    try
    {
        throw;
    } catch (UCXXConfigError& e) {
        PyErr_SetString(ucxx_config_error, e.what());
    } catch (UCXXError& e) {
        PyErr_SetString(ucxx_error, e.what());
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

}  // namespace ucxx
