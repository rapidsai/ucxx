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

PyObject* UCXXNoMessageError;
PyObject* UCXXNoResourceError;
PyObject* UCXXIOError;
PyObject* UCXXNoMemoryError;
PyObject* UCXXInvalidParamError;
PyObject* UCXXUnreachableError;
PyObject* UCXXInvalidAddrError;
PyObject* UCXXNotImplementedError;
PyObject* UCXXMessageTruncatedError;
PyObject* UCXXNoProgressError;
PyObject* UCXXBufferTooSmallError;
PyObject* UCXXNoElemError;
PyObject* UCXXSomeConnectsFailedError;
PyObject* UCXXNoDeviceError;
PyObject* UCXXBusyError;
PyObject* UCXXCanceledError;
PyObject* UCXXShmemSegmentError;
PyObject* UCXXAlreadyExistsError;
PyObject* UCXXOutOfRangeError;
PyObject* UCXXTimedOutError;
PyObject* UCXXExceedsLimitError;
PyObject* UCXXUnsupportedError;
PyObject* UCXXRejectedError;
PyObject* UCXXNotConnectedError;
PyObject* UCXXConnectionResetError;
PyObject* UCXXFirstLinkFailureError;
PyObject* UCXXLastLinkFailureError;
PyObject* UCXXFirstEndpointFailureError;
PyObject* UCXXEndpointTimeoutError;
PyObject* UCXXLastEndpointFailureError;

PyObject* UCXXCloseError;
PyObject* UCXXConfigError;

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

  new_exception(&UCXXNoMessageError, "UCXXNoMessageError", UCXXBaseException);
  new_exception(&UCXXNoResourceError, "UCXXNoResourceError", UCXXBaseException);
  new_exception(&UCXXIOError, "UCXXIOError", UCXXBaseException);
  new_exception(&UCXXNoMemoryError, "UCXXNoMemoryError", UCXXBaseException);
  new_exception(&UCXXInvalidParamError, "UCXXInvalidParamError", UCXXBaseException);
  new_exception(&UCXXUnreachableError, "UCXXUnreachableError", UCXXBaseException);
  new_exception(&UCXXInvalidAddrError, "UCXXInvalidAddrError", UCXXBaseException);
  new_exception(&UCXXNotImplementedError, "UCXXNotImplementedError", UCXXBaseException);
  new_exception(&UCXXMessageTruncatedError, "UCXXMessageTruncatedError", UCXXBaseException);
  new_exception(&UCXXNoProgressError, "UCXXNoProgressError", UCXXBaseException);
  new_exception(&UCXXBufferTooSmallError, "UCXXBufferTooSmallError", UCXXBaseException);
  new_exception(&UCXXNoElemError, "UCXXNoElemError", UCXXBaseException);
  new_exception(&UCXXSomeConnectsFailedError, "UCXXSomeConnectsFailedError", UCXXBaseException);
  new_exception(&UCXXNoDeviceError, "UCXXNoDeviceError", UCXXBaseException);
  new_exception(&UCXXBusyError, "UCXXBusyError", UCXXBaseException);
  new_exception(&UCXXCanceledError, "UCXXCanceledError", UCXXBaseException);
  new_exception(&UCXXShmemSegmentError, "UCXXShmemSegmentError", UCXXBaseException);
  new_exception(&UCXXAlreadyExistsError, "UCXXAlreadyExistsError", UCXXBaseException);
  new_exception(&UCXXOutOfRangeError, "UCXXOutOfRangeError", UCXXBaseException);
  new_exception(&UCXXTimedOutError, "UCXXTimedOutError", UCXXBaseException);
  new_exception(&UCXXExceedsLimitError, "UCXXExceedsLimitError", UCXXBaseException);
  new_exception(&UCXXUnsupportedError, "UCXXUnsupportedError", UCXXBaseException);
  new_exception(&UCXXRejectedError, "UCXXRejectedError", UCXXBaseException);
  new_exception(&UCXXNotConnectedError, "UCXXNotConnectedError", UCXXBaseException);
  new_exception(&UCXXConnectionResetError, "UCXXConnectionResetError", UCXXBaseException);
  new_exception(&UCXXFirstLinkFailureError, "UCXXFirstLinkFailureError", UCXXBaseException);
  new_exception(&UCXXLastLinkFailureError, "UCXXLastLinkFailureError", UCXXBaseException);
  new_exception(&UCXXFirstEndpointFailureError, "UCXXFirstEndpointFailureError", UCXXBaseException);
  new_exception(&UCXXEndpointTimeoutError, "UCXXEndpointTimeoutError", UCXXBaseException);
  new_exception(&UCXXLastEndpointFailureError, "UCXXLastEndpointFailureError", UCXXBaseException);

  new_exception(&UCXXConfigError, "UCXXConfigError", UCXXBaseException);
  new_exception(&UCXXCloseError, "UCXXCloseError", UCXXBaseException);
}

void raise_py_error()
{
  try {
    throw;
  } catch (const NoMessageError& e) {
    PyErr_SetString(UCXXNoMessageError, e.what());
  } catch (const NoResourceError& e) {
    PyErr_SetString(UCXXNoResourceError, e.what());
  } catch (const IOError& e) {
    PyErr_SetString(UCXXIOError, e.what());
  } catch (const NoMemoryError& e) {
    PyErr_SetString(UCXXNoMemoryError, e.what());
  } catch (const InvalidParamError& e) {
    PyErr_SetString(UCXXInvalidParamError, e.what());
  } catch (const UnreachableError& e) {
    PyErr_SetString(UCXXUnreachableError, e.what());
  } catch (const InvalidAddrError& e) {
    PyErr_SetString(UCXXInvalidAddrError, e.what());
  } catch (const NotImplementedError& e) {
    PyErr_SetString(UCXXNotImplementedError, e.what());
  } catch (const MessageTruncatedError& e) {
    PyErr_SetString(UCXXMessageTruncatedError, e.what());
  } catch (const NoProgressError& e) {
    PyErr_SetString(UCXXNoProgressError, e.what());
  } catch (const BufferTooSmallError& e) {
    PyErr_SetString(UCXXBufferTooSmallError, e.what());
  } catch (const NoElemError& e) {
    PyErr_SetString(UCXXNoElemError, e.what());
  } catch (const SomeConnectsFailedError& e) {
    PyErr_SetString(UCXXSomeConnectsFailedError, e.what());
  } catch (const NoDeviceError& e) {
    PyErr_SetString(UCXXNoDeviceError, e.what());
  } catch (const BusyError& e) {
    PyErr_SetString(UCXXBusyError, e.what());
  } catch (const CanceledError& e) {
    PyErr_SetString(UCXXCanceledError, e.what());
  } catch (const ShmemSegmentError& e) {
    PyErr_SetString(UCXXShmemSegmentError, e.what());
  } catch (const AlreadyExistsError& e) {
    PyErr_SetString(UCXXAlreadyExistsError, e.what());
  } catch (const OutOfRangeError& e) {
    PyErr_SetString(UCXXOutOfRangeError, e.what());
  } catch (const TimedOutError& e) {
    PyErr_SetString(UCXXTimedOutError, e.what());
  } catch (const ExceedsLimitError& e) {
    PyErr_SetString(UCXXExceedsLimitError, e.what());
  } catch (const UnsupportedError& e) {
    PyErr_SetString(UCXXUnsupportedError, e.what());
  } catch (const RejectedError& e) {
    PyErr_SetString(UCXXRejectedError, e.what());
  } catch (const NotConnectedError& e) {
    PyErr_SetString(UCXXNotConnectedError, e.what());
  } catch (const ConnectionResetError& e) {
    PyErr_SetString(UCXXConnectionResetError, e.what());
  } catch (const FirstLinkFailureError& e) {
    PyErr_SetString(UCXXFirstLinkFailureError, e.what());
  } catch (const LastLinkFailureError& e) {
    PyErr_SetString(UCXXLastLinkFailureError, e.what());
  } catch (const FirstEndpointFailureError& e) {
    PyErr_SetString(UCXXFirstEndpointFailureError, e.what());
  } catch (const EndpointTimeoutError& e) {
    PyErr_SetString(UCXXEndpointTimeoutError, e.what());
  } catch (const LastEndpointFailureError& e) {
    PyErr_SetString(UCXXLastEndpointFailureError, e.what());
  } catch (const CloseError& e) {
    PyErr_SetString(UCXXCloseError, e.what());
  } catch (const ConfigError& e) {
    PyErr_SetString(UCXXConfigError, e.what());
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
