/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <Python.h>

#include <ucp/api/ucp.h>

#include <ucxx/log.h>
#include <ucxx/python/exception.h>
#include <ucxx/python/python_future.h>

namespace ucxx {

namespace python {

Future::Future(std::shared_ptr<::ucxx::Notifier> notifier) : ::ucxx::Future(notifier) {}

std::shared_ptr<::ucxx::Future> createFuture(std::shared_ptr<::ucxx::Notifier> notifier)
{
  return std::shared_ptr<::ucxx::Future>(new ::ucxx::python::Future(notifier));
}

Future::~Future()
{
  // TODO: check it is truly safe to require the GIL here. Segfaults can occur
  // if `Py_XDECREF` is called but the thread doesn't currently own the GIL.
  PyGILState_STATE state = PyGILState_Ensure();
  Py_XDECREF(_handle);
  PyGILState_Release(state);
}

void Future::set(ucs_status_t status)
{
  if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");

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
  if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");

  auto s = shared_from_this();

  ucxx_trace_req("Future::notify() this: %p, shared.get(): %p, handle: %p, notifier: %p",
                 this,
                 s.get(),
                 _handle,
                 _notifier.get());
  _notifier->scheduleFutureNotify(shared_from_this(), status);
}

void* Future::getHandle()
{
  if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");

  return _handle;
}

void* Future::release()
{
  if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");

  return std::exchange(_handle, nullptr);
}

}  // namespace python

}  // namespace ucxx
