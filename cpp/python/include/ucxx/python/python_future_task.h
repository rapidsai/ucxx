/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <chrono>
#include <future>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <iostream>

#include <Python.h>

#include <ucxx/log.h>
#include <ucxx/python/future.h>

namespace ucxx {

namespace python {

template <typename ReturnType, typename... TaskArgs>
class PythonFutureTask : public std::enable_shared_from_this<PythonFutureTask<ReturnType>> {
 private:
  std::packaged_task<ReturnType(TaskArgs...)> _task{};  ///< The user-defined C++ task to run
  std::function<PyObject*(ReturnType)>
    _pythonConvert{};                 ///< Function to convert the C++ result into Python value
  PyObject* _asyncioEventLoop{};      ///< The handle to the Python asyncio event loop
  PyObject* _handle{};                ///< The handle to the Python future
  std::future<ReturnType> _future{};  ///< The C++ future containing the task result

  /**
   * @brief Set the result value of the Python future.
   *
   * Set the result value of the underlying Python future using the `pythonConvert` function
   * specified in the constructor to convert the C++ result into the `PyObject*`.
   *
   * This function will take the GIL to convert the C++ result into the `PyObject*`.
   *
   * @param[in] result the C++ value that will be converted and set as the result of the
   *                   Python future.
   */
  void setResult(const ReturnType result)
  {
    // PyLong_FromSize_t requires the GIL
    if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");
    PyGILState_STATE state = PyGILState_Ensure();
    ucxx::python::future_set_result(_asyncioEventLoop, _handle, _pythonConvert(result));
    PyGILState_Release(state);
  }

  /**
   * @brief Set the exception of the Python future.
   *
   * Set the exception of the underlying Python future. Currently any exceptions that the
   * task may raise must be derived from `std::exception`.
   *
   * @param[in] pythonException the Python exception type to raise.
   * @param[in] message the message of the exception.
   */
  void setPythonException(PyObject* pythonException, const std::string& message)
  {
    if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");
    ucxx::python::future_set_exception(
      _asyncioEventLoop, _handle, pythonException, message.c_str());
  }

  /**
   * @brief Parse C++ exception as a Python exception and set the Python future exception.
   *
   * Parse a C++ exception as a Python exception and set the Python future exception.
   * Currently any exceptions that the task may raise must be derived from `std::exception`.
   *
   * @param[in] exception the C++ exception that was raised by the user-defined task.
   */
  void setException(const std::exception& exception)
  {
    try {
      throw exception;
    } catch (const std::bad_alloc& e) {
      setPythonException(PyExc_MemoryError, e.what());
    } catch (const std::bad_cast& e) {
      setPythonException(PyExc_TypeError, e.what());
    } catch (const std::bad_typeid& e) {
      setPythonException(PyExc_TypeError, e.what());
    } catch (const std::domain_error& e) {
      setPythonException(PyExc_ValueError, e.what());
    } catch (const std::invalid_argument& e) {
      setPythonException(PyExc_ValueError, e.what());
    } catch (const std::ios_base::failure& e) {
      setPythonException(PyExc_IOError, e.what());
    } catch (const std::out_of_range& e) {
      setPythonException(PyExc_IndexError, e.what());
    } catch (const std::overflow_error& e) {
      setPythonException(PyExc_OverflowError, e.what());
    } catch (const std::range_error& e) {
      setPythonException(PyExc_ArithmeticError, e.what());
    } catch (const std::underflow_error& e) {
      setPythonException(PyExc_ArithmeticError, e.what());
    } catch (const std::exception& e) {
      setPythonException(PyExc_RuntimeError, e.what());
    } catch (...) {
      setPythonException(PyExc_RuntimeError, "Unknown exception");
    }
  }

 public:
  /**
   * @brief Construct a Python future backed by C++ `std::packaged_task`.
   *
   * Construct a future object that receives a user-defined C++ `std::packaged_task` which
   * runs asynchronously using an internal `std::async` that ultimately notifies a Python
   * future that can be awaited in Python code.
   *
   * @param[in] task the user-defined C++ task.
   * @param[in] pythonConvert C-Python function to convert a C object into a `PyObject*`
   *                          representing the result of the task.
   * @param[in] asyncioEventLoop pointer to a valid Python object containing the event loop
   *                             that the application is using, to which the Python future
   *                             will belong to.
   * @param[in] launchPolicy launch policy for the async C++ task.
   */
  explicit PythonFutureTask(std::packaged_task<ReturnType(TaskArgs...)> task,
                            std::function<PyObject*(ReturnType)> pythonConvert,
                            PyObject* asyncioEventLoop,
                            std::launch launchPolicy = std::launch::async)
    : _task{std::move(task)},
      _pythonConvert(pythonConvert),
      _asyncioEventLoop(asyncioEventLoop),
      _handle{ucxx::python::create_python_future(asyncioEventLoop)}
  {
    _future = std::async(launchPolicy, [this]() {
      std::future<ReturnType> result = this->_task.get_future();
      this->_task();
      try {
        const ReturnType r = result.get();
        this->setException(r);
        return r;
      } catch (std::exception& e) {
        this->setException(e);
      }
    });
  }
  PythonFutureTask(const PythonFutureTask&)            = delete;
  PythonFutureTask& operator=(PythonFutureTask const&) = delete;
  PythonFutureTask(PythonFutureTask&& o)               = delete;
  PythonFutureTask& operator=(PythonFutureTask&& o)    = delete;

  /**
   * @brief Python future destructor.
   *
   * Decrement the reference count of the underlying Python future.
   */
  ~PythonFutureTask() { Py_XDECREF(_handle); }

  /**
   * @brief Get the C++ future.
   *
   * Get the underlying C++ future that can be awaited and have its result read.
   *
   * @returns The underlying C++ future.
   */
  std::future<ReturnType>& getFuture() { return _future; }

  /**
   * @brief Get the underlying future `PyObject*` handle but does not release ownership.
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
  PyObject* getHandle()
  {
    if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");

    return _handle;
  }

  /**
   * @brief Get the underlying future `PyObject*` handle and release ownership.
   *
   * Get the underlying `PyObject*` handle releasing ownership. This should be used when
   * the future needs to be permanently transferred to Python code. After calling this
   * method the object becomes invalid for any other uses.
   *
   * @throws std::runtime_error if the object is invalid or has been already released.
   *
   * @returns The underlying `PyObject*` handle.
   */
  PyObject* release()
  {
    if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");

    return std::exchange(_handle, nullptr);
  }
};

}  // namespace python

}  // namespace ucxx
