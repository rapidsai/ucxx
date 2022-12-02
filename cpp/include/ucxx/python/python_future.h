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
#include <ucxx/python/notifier.h>

namespace ucxx {

namespace python {

class Future : public std::enable_shared_from_this<Future> {
 private:
  PyObject* _handle{create_python_future()};
  std::shared_ptr<Notifier> _notifier{};

 public:
  Future()              = delete;
  Future(const Future&) = delete;
  Future& operator=(Future const&) = delete;
  Future(Future&& o)               = delete;
  Future& operator=(Future&& o) = delete;

  Future(std::shared_ptr<Notifier> notifier);

  void notify(ucs_status_t status);

  void set(ucs_status_t status);

  PyObject* getHandle();
};

}  // namespace python

}  // namespace ucxx

#endif
