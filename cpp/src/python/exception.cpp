/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#if UCXX_ENABLE_PYTHON
#include <ios>
#include <stdexcept>

#include <ucxx/exception.h>
#include <ucxx/python/exception.h>

namespace ucxx {

namespace python {

PyObject* UCXXBaseException;
PyObject* UCXXError;
PyObject* UCXXCanceledError;
PyObject* UCXXCloseError;
PyObject* UCXXConfigError;
PyObject* UCXXConnectionResetError;
PyObject* UCXXMessageTruncatedError;

static PyObject* new_exception(PyObject** exception, const char* name, PyObject* base)
{
  char qualified_name[255];

  if (*exception == NULL) {
    sprintf(qualified_name, "ucxx.%s", name);

    *exception = PyErr_NewException(qualified_name, base, NULL);
  }
  return *exception;
}

void create_exceptions()
{
  new_exception(&UCXXBaseException, "UCXXBaseException", NULL);
  new_exception(&UCXXError, "UCXXError", UCXXBaseException);
  new_exception(&UCXXCanceledError, "UCXXCanceledError", UCXXBaseException);
  new_exception(&UCXXConfigError, "UCXXConfigError", UCXXBaseException);
  new_exception(&UCXXCloseError, "UCXXCloseError", UCXXBaseException);
  new_exception(&UCXXConnectionResetError, "UCXXConnectionResetError", UCXXBaseException);
  new_exception(&UCXXMessageTruncatedError, "UCXXMessageTruncatedError", UCXXBaseException);
}

void raise_py_error()
{
  try {
    throw;
  } catch (const CanceledError& e) {
    PyErr_SetString(UCXXCanceledError, e.what());
  } catch (const ConfigError& e) {
    PyErr_SetString(UCXXConfigError, e.what());
  } catch (const ConnectionResetError& e) {
    PyErr_SetString(UCXXConnectionResetError, e.what());
  } catch (const MessageTruncatedError& e) {
    PyErr_SetString(UCXXMessageTruncatedError, e.what());
  } catch (const Error& e) {
    PyErr_SetString(UCXXError, e.what());
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
    case UCS_ERR_CANCELED: return UCXXCanceledError;
    case UCS_ERR_CONNECTION_RESET: return UCXXConnectionResetError;
    case UCS_ERR_MESSAGE_TRUNCATED: return UCXXMessageTruncatedError;
    default: return UCXXError;
  }
}

}  // namespace python

}  // namespace ucxx

#endif
