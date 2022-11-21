/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#if UCXX_ENABLE_PYTHON
#include <Python.h>

#include <ucp/api/ucp.h>

#include <ucxx/log.h>
#include <ucxx/python/python_future.h>

namespace ucxx {

namespace python {

void Future::set(ucs_status_t status)
{
  ucxx_trace_req(
    "Future::set() this: %p, _handle: %p, status: %s", this, _handle, ucs_status_string(status));
  if (status == UCS_OK)
    future_set_result(_handle, Py_True);
  else
    future_set_exception(
      _handle, get_python_exception_from_ucs_status(status), ucs_status_string(status));
}

void Future::notify(ucs_status_t status)
{
  auto s = shared_from_this();

  ucxx_trace_req("Future::notify() this: %p, shared.get(): %p, handle: %p, notifier: %p",
                 this,
                 s.get(),
                 _handle,
                 _notifier.get());
  _notifier->scheduleFutureNotify(shared_from_this(), status);
}

}  // namespace python

}  // namespace ucxx

#endif
