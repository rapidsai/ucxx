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

class UCXXRequest : public UCXXComponent {
 protected:
  std::shared_ptr<ucxx_request_t> _handle{nullptr};
  std::shared_ptr<UCXXEndpoint> _endpoint{nullptr};
  std::shared_ptr<NotificationRequest> _notificationRequest{nullptr};
  std::string _operationName{"request_undefined"};
  ucs_status_ptr_t _requestStatusPtr{nullptr};
  ucs_status_t _requestStatus{UCS_INPROGRESS};

  UCXXRequest(std::shared_ptr<UCXXEndpoint> endpoint,
              std::shared_ptr<NotificationRequest> notificationRequest,
              const std::string operationName);

  void process();

  void setStatus();

 public:
  UCXXRequest()                   = delete;
  UCXXRequest(const UCXXRequest&) = delete;
  UCXXRequest& operator=(UCXXRequest const&) = delete;
  UCXXRequest(UCXXRequest&& o)               = delete;
  UCXXRequest& operator=(UCXXRequest&& o) = delete;

  ~UCXXRequest();

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
