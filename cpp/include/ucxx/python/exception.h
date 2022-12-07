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

extern PyObject* UCXXNoMessageError;
extern PyObject* UCXXNoResourceError;
extern PyObject* UCXXIOError;
extern PyObject* UCXXNoMemoryError;
extern PyObject* UCXXInvalidParamError;
extern PyObject* UCXXUnreachableError;
extern PyObject* UCXXInvalidAddrError;
extern PyObject* UCXXNotImplementedError;
extern PyObject* UCXXMessageTruncatedError;
extern PyObject* UCXXNoProgressError;
extern PyObject* UCXXBufferTooSmallError;
extern PyObject* UCXXNoElemError;
extern PyObject* UCXXSomeConnectsFailedError;
extern PyObject* UCXXNoDeviceError;
extern PyObject* UCXXBusyError;
extern PyObject* UCXXCanceledError;
extern PyObject* UCXXShmemSegmentError;
extern PyObject* UCXXAlreadyExistsError;
extern PyObject* UCXXOutOfRangeError;
extern PyObject* UCXXTimedOutError;
extern PyObject* UCXXExceedsLimitError;
extern PyObject* UCXXUnsupportedError;
extern PyObject* UCXXRejectedError;
extern PyObject* UCXXNotConnectedError;
extern PyObject* UCXXConnectionResetError;
extern PyObject* UCXXFirstLinkFailureError;
extern PyObject* UCXXLastLinkFailureError;
extern PyObject* UCXXFirstEndpointFailureError;
extern PyObject* UCXXEndpointTimeoutError;
extern PyObject* UCXXLastEndpointFailureError;

extern PyObject* UCXXCloseError;
extern PyObject* UCXXConfigError;

void create_exceptions();

void raise_py_error();

PyObject* get_python_exception_from_ucs_status(ucs_status_t status);

}  // namespace python

}  // namespace ucxx

#endif
