/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <mutex>
#include <vector>

#include <Python.h>

namespace ucxx {

namespace python {

class PythonFutureTaskCollector {
 public:
  std::vector<PyObject*> _toCollect{};  ///< Tasks to be collected
  std::mutex _mutex{};                  ///< Mutex to provide safe access to `_toCollect`.

  /**
   * Get reference to `PythonFutureTaskCollector` instance.
   *
   * `PythonFutureTaskCollector` is a singleton and thus must not be directly instantiated.
   * Instead, users should call this static method to get a reference to its instance.
   */
  static PythonFutureTaskCollector& get();

  /**
   * Push a Python handle to be later collected.
   *
   * Push a Python handle to be later collected. This method does not require the GIL.
   */
  void push(PyObject* handle);

  /**
   * Decrement each reference previously pushed exactly once.
   *
   * Decrement each reference (i.e., garbage collect) that was previously pushed via the
   * `push()` method exactly once, cleaning internal references at the end.
   *
   * WARNING: Calling this method will attempt to take the GIL, so make sure no other thread
   * currently owns it while this thread also competes for other resources that the Python
   * thread holding the GIL may require as it may cause a deadlock.
   */
  void collect();

 private:
  /**
   * Private constructor.
   *
   * Private constructor to prevent accidental instantiation of multiple collectors.
   */
  PythonFutureTaskCollector();

 public:
  PythonFutureTaskCollector(const PythonFutureTaskCollector&)            = delete;
  PythonFutureTaskCollector& operator=(PythonFutureTaskCollector const&) = delete;
  PythonFutureTaskCollector(PythonFutureTaskCollector&& o)               = delete;
  PythonFutureTaskCollector& operator=(PythonFutureTaskCollector&& o)    = delete;

  /**
   * Destructor.
   *
   * Destructor of the collector. Warns if any tasks were pushed but not collected.
   */
  ~PythonFutureTaskCollector();
};

}  // namespace python

}  // namespace ucxx
