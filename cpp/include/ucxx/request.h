/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

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
  std::shared_ptr<ucxx_request_t> _handle{nullptr};
  std::shared_ptr<Endpoint> _endpoint{nullptr};
  std::shared_ptr<NotificationRequest> _notificationRequest{nullptr};
  std::string _operationName{"request_undefined"};
  ucs_status_ptr_t _requestStatusPtr{nullptr};
  ucs_status_t _requestStatus{UCS_INPROGRESS};
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

  std::shared_ptr<ucxx_request_t> getHandle();

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
