/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <atomic>
#include <chrono>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint.h>
#include <ucxx/typedefs.h>

#if UCXX_ENABLE_PYTHON
#include <Python.h>
#else
typedef void PyObject;
#endif

namespace ucxx {

class Request : public Component {
 protected:
  std::atomic<ucs_status_t> _status{UCS_INPROGRESS};
  std::string _status_msg{};
  void* _request{nullptr};
#if UCXX_ENABLE_PYTHON
  std::shared_ptr<python::Future> _pythonFuture{nullptr};
#endif
  std::function<void(std::shared_ptr<void>)> _callback{nullptr};
  std::shared_ptr<void> _callbackData{nullptr};
  std::shared_ptr<Worker> _worker{nullptr};
  std::shared_ptr<Endpoint> _endpoint{nullptr};
  std::shared_ptr<DelayedSubmission> _delayedSubmission{nullptr};
  std::string _operationName{"request_undefined"};
  bool _enablePythonFuture{true};

  Request(std::shared_ptr<Component> endpointOrWorker,
          std::shared_ptr<DelayedSubmission> delayedSubmission,
          const std::string operationName,
          const bool enablePythonFuture = false);

  void process();

  void setStatus(ucs_status_t status);

 public:
  Request()               = delete;
  Request(const Request&) = delete;
  Request& operator=(Request const&) = delete;
  Request(Request&& o)               = delete;
  Request& operator=(Request&& o) = delete;

  virtual ~Request();

  void cancel();

  ucs_status_t getStatus();

  PyObject* getPyFuture();

  void checkError();

  bool isCompleted();

  void callback(void* request, ucs_status_t status);

  virtual void populateDelayedSubmission() = 0;
};

}  // namespace ucxx
