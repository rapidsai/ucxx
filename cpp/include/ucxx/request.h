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
#endif

namespace ucxx {

class Request : public Component {
 protected:
  std::atomic<ucs_status_t> _status{UCS_INPROGRESS};
  void* _request{nullptr};
#if UCXX_ENABLE_PYTHON
  std::shared_ptr<python::Future> _pythonFuture{nullptr};
#endif
  std::function<void(std::shared_ptr<void>)> _callback{nullptr};
  std::shared_ptr<void> _callbackData{nullptr};
  std::shared_ptr<Endpoint> _endpoint{nullptr};
  std::shared_ptr<NotificationRequest> _notificationRequest{nullptr};
  std::string _operationName{"request_undefined"};
  bool _enablePythonFuture{true};

  Request(std::shared_ptr<Endpoint> endpoint,
          std::shared_ptr<NotificationRequest> notificationRequest,
          const std::string operationName,
          const bool enablePythonFuture = true);

  void process();

  void setStatus();

 public:
  Request()               = delete;
  Request(const Request&) = delete;
  Request& operator=(Request const&) = delete;
  Request(Request&& o)               = delete;
  Request& operator=(Request&& o) = delete;

  ~Request();

  void cancel();

  ucs_status_t getStatus();

  PyObject* getPyFuture();

  void checkError();

  template <typename Rep, typename Period>
  bool isCompleted(std::chrono::duration<Rep, Period> period);

  bool isCompleted(int64_t periodNs = 0);

  void callback(void* request, ucs_status_t status);

  virtual void populateNotificationRequest() = 0;
};

}  // namespace ucxx
