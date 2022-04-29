/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#ifdef UCXX_ENABLE_PYTHON
#include <Python.h>

#include <ucp/api/ucp.h>

#include <ucxx/log.h>
#include <ucxx/python/python_future.h>

namespace ucxx {

void PythonFuture::set(ucs_status_t status)
{
  ucxx_trace_req("PythonFuture::set() this: %p, _handle: %p, status: %s",
                 this,
                 _handle,
                 ucs_status_string(status));
  if (status == UCS_OK)
    future_set_result(_handle, Py_True);
  else
    future_set_exception(
      _handle, ucxx::get_python_exception_from_ucs_status(status), ucs_status_string(status));
}

void PythonFuture::notify(ucs_status_t status)
{
  auto s = shared_from_this();

  ucxx_trace_req("PythonFuture::notify() this: %p, shared.get(): %p, handle: %p, notifier: %p",
                 this,
                 s.get(),
                 _handle,
                 _notifier.get());
  _notifier->schedulePythonFutureNotify(shared_from_this(), status);
}

}  // namespace ucxx

#endif
