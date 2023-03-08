/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <functional>
#include <memory>

#include <Python.h>

#include <ucp/api/ucp.h>

#include <ucxx/future.h>
#include <ucxx/notifier.h>
#include <ucxx/python/future.h>

namespace ucxx {

namespace python {

class Future : public ::ucxx::Future {
 private:
  PyObject* _handle{create_python_future()};  ///< The handle to the Python future

  /**
   * @brief Construct a future that may be notified from a notifier thread.
   *
   * Construct a future that may be notified from a notifier running on its own thread and
   * thus will decrease overhead from the application thread.
   *
   * This class may also be used to set the result or exception from any thread, but that
   * currently requires explicitly taking the GIL before calling `set()`.
   *
   * @param[in] notifier  notifier object running on a separate thread.
   */
  explicit Future(std::shared_ptr<::ucxx::Notifier> notifier);

 public:
  Future()              = delete;
  Future(const Future&) = delete;
  Future& operator=(Future const&) = delete;
  Future(Future&& o)               = delete;
  Future& operator=(Future&& o) = delete;

  friend std::shared_ptr<::ucxx::Future> createPythonFuture(
    std::shared_ptr<::ucxx::Notifier> notifier);

  /**
   * @brief Virtual destructor.
   *
   * Virtual destructor with empty implementation.
   */
  virtual ~Future();

  /**
   * @brief Inform the notifier thread that the future has completed.
   *
   * Inform the notifier thread that the future has completed so it can notify the event
   * loop of that occurrence.
   *
   * @throws std::runtime_error if the object is invalid or has been already released.
   *
   * @param[in] status  request completion status.
   */
  void notify(ucs_status_t status);

  /**
   * @brief Set the future completion status.
   *
   * Set the future status as completed, either with a successful completion or error.
   *
   * @throws std::runtime_error if the object is invalid or has been already released.
   *
   * @param[in] status  request completion status.
   */
  void set(ucs_status_t status);

  /**
   * @brief Get the underlying `PyObject*` handle but does not release ownership.
   *
   * Get the underlying `PyObject*` handle without releasing ownership. This can be useful
   * for example for logging, where we want to see the address of the pointer but do not
   * want to transfer ownership.
   *
   * @warning The destructor will also destroy the Python future, a pointer taken via this
   * method will cause the object to become invalid.
   *
   * @throws std::runtime_error if the object is invalid or has been already released.
   *
   * @returns The underlying `PyObject*` handle.
   */
  void* getHandle();

  /**
   * @brief Get the underlying `PyObject*` handle and release ownership.
   *
   * Get the underlying `PyObject*` handle releasing ownership. This should be used when
   * the future needs to be permanently transferred to Python code. After calling this
   * method the object becomes invalid for any other uses.
   *
   * @throws std::runtime_error if the object is invalid or has been already released.
   *
   * @returns The underlying `PyObject*` handle.
   */
  void* release();
};

}  // namespace python

}  // namespace ucxx
