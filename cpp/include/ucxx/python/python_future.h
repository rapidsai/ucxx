/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#if UCXX_ENABLE_PYTHON
#include <functional>
#include <memory>

#include <Python.h>

#include <ucp/api/ucp.h>

#include <ucxx/log.h>
#include <ucxx/python/future.h>

namespace ucxx {

namespace python {

class Notifier;

typedef void (*FutureCallback)();

class Future : public std::enable_shared_from_this<Future> {
 private:
  PyObject* _handle{create_python_future()};
  std::shared_ptr<Notifier> _notifier{};
  void* _worker;

 public:
  Future(std::shared_ptr<Notifier> notifier) : _notifier(notifier) {}

  void notify(ucs_status_t status);

  void set(ucs_status_t status);

  PyObject* getHandle() { return _handle; }
};

}  // namespace python

}  // namespace ucxx

#endif
