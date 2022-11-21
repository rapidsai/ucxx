/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#if UCXX_ENABLE_PYTHON
#include <exception>

#include <ucxx/exception.h>
#include <ucxx/python/exception.h>

namespace ucxx {

namespace python {

void raise_py_error()
{
  try {
    throw;
  } catch (const CanceledError& e) {
    PyErr_SetString(ucxx_canceled_error, e.what());
  } catch (const ConfigError& e) {
    PyErr_SetString(ucxx_config_error, e.what());
  } catch (const ConnectionResetError& e) {
    PyErr_SetString(ucxx_connection_reset_error, e.what());
  } catch (const MessageTruncatedError& e) {
    PyErr_SetString(ucxx_message_truncated_error, e.what());
  } catch (const Error& e) {
    PyErr_SetString(ucxx_error, e.what());
  } catch (const std::bad_alloc& e) {
    PyErr_SetString(PyExc_MemoryError, e.what());
  } catch (const std::bad_cast& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
  } catch (const std::bad_typeid& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
  } catch (const std::domain_error& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
  } catch (const std::invalid_argument& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
  } catch (const std::ios_base::failure& e) {
    PyErr_SetString(PyExc_IOError, e.what());
  } catch (const std::out_of_range& e) {
    PyErr_SetString(PyExc_IndexError, e.what());
  } catch (const std::overflow_error& e) {
    PyErr_SetString(PyExc_OverflowError, e.what());
  } catch (const std::range_error& e) {
    PyErr_SetString(PyExc_ArithmeticError, e.what());
  } catch (const std::underflow_error& e) {
    PyErr_SetString(PyExc_ArithmeticError, e.what());
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
  }
}

PyObject* get_python_exception_from_ucs_status(ucs_status_t status)
{
  switch (status) {
    case UCS_ERR_CANCELED: return ucxx_canceled_error;
    case UCS_ERR_CONNECTION_RESET: return ucxx_connection_reset_error;
    case UCS_ERR_MESSAGE_TRUNCATED: return ucxx_message_truncated_error;
    default: return ucxx_error;
  }
}

}  // namespace python

}  // namespace ucxx

#endif
