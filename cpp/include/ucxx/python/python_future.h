/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#ifdef UCXX_ENABLE_PYTHON
#include <functional>
#include <memory>

#include <Python.h>

#include <ucp/api/ucp.h>

#include <ucxx/log.h>
#include <ucxx/python/future.h>

namespace ucxx {

class UCXXNotifier;

typedef void (*PythonFutureCallback)();

class PythonFuture : public std::enable_shared_from_this<PythonFuture> {
 private:
  PyObject* _handle{create_python_future()};
  std::shared_ptr<ucxx::UCXXNotifier> _notifier{};
  void* _worker;

 public:
  PythonFuture(std::shared_ptr<ucxx::UCXXNotifier> notifier) : _notifier(notifier) {}

  void notify(ucs_status_t status);

  void set(ucs_status_t status);

  PyObject* getHandle() { return _handle; }
};

}  // namespace ucxx

#endif
