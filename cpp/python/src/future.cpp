/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <ucxx/log.h>

#include <Python.h>

namespace ucxx {

namespace python {

PyObject* asyncio_str              = NULL;
PyObject* asyncio_future_object    = NULL;
PyObject* call_soon_threadsafe_str = NULL;
PyObject* create_future_str        = NULL;
PyObject* future_str               = NULL;
PyObject* set_exception_str        = NULL;
PyObject* set_result_str           = NULL;

static int intern_strings(void)
{
  asyncio_str = PyUnicode_InternFromString("asyncio");
  if (asyncio_str == NULL) { return -1; }
  call_soon_threadsafe_str = PyUnicode_InternFromString("call_soon_threadsafe");
  if (call_soon_threadsafe_str == NULL) { return -1; }
  create_future_str = PyUnicode_InternFromString("create_future");
  if (create_future_str == NULL) { return -1; }
  future_str = PyUnicode_InternFromString("Future");
  if (future_str == NULL) { return -1; }
  set_exception_str = PyUnicode_InternFromString("set_exception");
  if (set_exception_str == NULL) { return -1; }
  set_result_str = PyUnicode_InternFromString("set_result");
  if (set_result_str == NULL) { return -1; }
  return 0;
}

static int init_ucxx_python()
{
  if (intern_strings() < 0) goto err;

  return 0;

err:
  if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not initialize  Python C-API.");
  return -1;
}

static PyObject* get_asyncio_future_object()
{
  PyObject* asyncio_module = NULL;

  if (asyncio_future_object) return asyncio_future_object;

  PyGILState_STATE state = PyGILState_Ensure();

  if (init_ucxx_python() < 0) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not allocate internals.");
    goto finish;
  }

  asyncio_module = PyImport_Import(asyncio_str);
  if (PyErr_Occurred()) ucxx_trace_req("Python error here");
  if (PyErr_Occurred()) PyErr_Print();
  if (asyncio_module == NULL) goto finish;

  asyncio_future_object = PyObject_GetAttr(asyncio_module, future_str);
  if (PyErr_Occurred()) ucxx_trace_req("Python error here");
  if (PyErr_Occurred()) PyErr_Print();
  Py_DECREF(asyncio_module);
  if (asyncio_future_object == NULL) { goto finish; }

finish:
  PyGILState_Release(state);
  return asyncio_future_object;
}

PyObject* create_python_future()
{
  PyObject* future_object = NULL;
  PyObject* result        = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  if (init_ucxx_python() < 0) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not allocate internals.");
    goto finish;
  }

  future_object = get_asyncio_future_object();
  if (future_object == NULL) { goto finish; }
  if (!PyCallable_Check(future_object)) {
    PyErr_Format(PyExc_RuntimeError,
                 "%s.%s is not callable.",
                 PyUnicode_1BYTE_DATA(asyncio_str),
                 PyUnicode_1BYTE_DATA(future_str));
    goto finish;
  }

  result = PyObject_CallFunctionObjArgs(future_object, NULL);
  if (PyErr_Occurred()) ucxx_trace_req("Python error here");
  if (PyErr_Occurred()) PyErr_Print();

finish:
  PyGILState_Release(state);
  return result;
}

static PyCFunction get_future_method(const char* method_name)
{
  PyCFunction result = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  PyObject* future_object = get_asyncio_future_object();
  if (PyErr_Occurred()) ucxx_trace_req("Python error here");
  if (PyErr_Occurred()) PyErr_Print();
  PyMethodDef* m = reinterpret_cast<PyTypeObject*>(future_object)->tp_methods;

  for (; m != NULL; ++m) {
    if (m->ml_name && !strcmp(m->ml_name, method_name)) {
      result = m->ml_meth;
      break;
    }
  }

  if (!result)
    PyErr_Format(PyExc_RuntimeError, "Unable to load function pointer for `Future.set_result`.");

  PyGILState_Release(state);
  return result;
}

PyObject* future_set_result(PyObject* future, PyObject* value)
{
  PyObject* result = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  PyCFunction f = get_future_method("set_result");
  result        = f(future, value);
  if (PyErr_Occurred()) ucxx_trace_req("Python error here");
  if (PyErr_Occurred()) PyErr_Print();

  PyGILState_Release(state);

  return result;
}

PyObject* future_set_exception(PyObject* future, PyObject* exception, const char* message)
{
  PyObject* result           = NULL;
  PyObject* message_object   = NULL;
  PyObject* message_tuple    = NULL;
  PyObject* formed_exception = NULL;
  PyCFunction f              = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  message_object = PyUnicode_FromString(message);
  if (message_object == NULL) goto err;
  message_tuple = PyTuple_Pack(1, message_object);
  if (message_tuple == NULL) goto err;
  formed_exception = PyObject_Call(exception, message_tuple, NULL);
  if (formed_exception == NULL) goto err;

  f = get_future_method("set_exception");

  result = f(future, formed_exception);
  goto finish;

err:
  PyErr_Format(PyExc_RuntimeError, "Error while setting exception for `asyncio.Future`.");
finish:
  Py_XDECREF(message_object);
  Py_XDECREF(message_tuple);
  Py_XDECREF(formed_exception);
  PyGILState_Release(state);
  return result;
}

PyObject* create_python_future_with_event_loop(PyObject* event_loop)
{
  PyObject* result = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  if (init_ucxx_python() < 0) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not allocate internals.");
    goto finish;
  }

  result = PyObject_CallMethodObjArgs(event_loop, create_future_str, NULL);
  if (PyErr_Occurred()) {
    ucxx_trace_req("Error calling event loop `create_future`.");
    PyErr_Print();
  }

finish:
  PyGILState_Release(state);
  return result;
}

PyObject* future_set_result_with_event_loop(PyObject* event_loop, PyObject* future, PyObject* value)
{
  PyObject* result              = NULL;
  PyObject* set_result_callable = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  if (init_ucxx_python() < 0) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not allocate internals.");
    goto finish;
  }

  set_result_callable = PyObject_GetAttr(future, set_result_str);
  if (PyErr_Occurred()) {
    ucxx_trace_req("Error getting future `set_result` method.");
    PyErr_Print();
    goto finish;
  }
  if (!PyCallable_Check(set_result_callable)) {
    PyErr_Format(PyExc_RuntimeError,
                 "%s.%s is not callable.",
                 PyUnicode_1BYTE_DATA(future),
                 PyUnicode_1BYTE_DATA(set_result_str));
    goto finish;
  }

  result = PyObject_CallMethodObjArgs(
    event_loop, call_soon_threadsafe_str, set_result_callable, value, NULL);
  if (PyErr_Occurred()) {
    ucxx_trace_req("Error calling `call_soon_threadsafe` to set future result.");
    PyErr_Print();
  }

finish:
  Py_XDECREF(set_result_callable);
  PyGILState_Release(state);
  return result;
}

PyObject* future_set_exception_with_event_loop(PyObject* event_loop,
                                               PyObject* future,
                                               PyObject* exception,
                                               const char* message)
{
  PyObject* result                 = NULL;
  PyObject* set_exception_callable = NULL;
  PyObject* message_object         = NULL;
  PyObject* message_tuple          = NULL;
  PyObject* formed_exception       = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  if (init_ucxx_python() < 0) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not allocate internals.");
    goto finish;
  }

  set_exception_callable = PyObject_GetAttr(future, set_exception_str);
  if (PyErr_Occurred()) {
    ucxx_trace_req("Error getting future `set_exception` method.");
    PyErr_Print();
    goto finish;
  }
  if (!PyCallable_Check(set_exception_callable)) {
    PyErr_Format(PyExc_RuntimeError,
                 "%s.%s is not callable.",
                 PyUnicode_1BYTE_DATA(future),
                 PyUnicode_1BYTE_DATA(set_exception_str));
    goto finish;
  }

  message_object = PyUnicode_FromString(message);
  if (message_object == NULL) goto err;
  message_tuple = PyTuple_Pack(1, message_object);
  if (message_tuple == NULL) goto err;
  formed_exception = PyObject_Call(exception, message_tuple, NULL);
  if (formed_exception == NULL) goto err;

  result = PyObject_CallMethodObjArgs(
    event_loop, call_soon_threadsafe_str, set_exception_callable, formed_exception, NULL);
  if (PyErr_Occurred()) {
    ucxx_trace_req("Error calling `call_soon_threadsafe` to set future exception.");
    PyErr_Print();
  }
  goto finish;

err:
  PyErr_Format(PyExc_RuntimeError,
               "Error while forming exception for `asyncio.Future.set_exception`.");
finish:
  Py_XDECREF(message_object);
  Py_XDECREF(message_tuple);
  Py_XDECREF(formed_exception);
  Py_XDECREF(set_exception_callable);
  PyGILState_Release(state);
  return result;
}

}  // namespace python

}  // namespace ucxx
