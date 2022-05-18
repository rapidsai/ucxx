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
  inflight_requests_t _inflight_requests{nullptr};

  UCXXRequest(std::shared_ptr<UCXXEndpoint> endpoint,
              inflight_requests_t inflight_requests,
              std::shared_ptr<ucxx_request_t> request);

 public:
  UCXXRequest()                   = delete;
  UCXXRequest(const UCXXRequest&) = delete;
  UCXXRequest& operator=(UCXXRequest const&) = delete;
  UCXXRequest(UCXXRequest&& o)               = delete;
  UCXXRequest& operator=(UCXXRequest&& o) = delete;

  ~UCXXRequest();

  friend std::shared_ptr<UCXXRequest> createRequest(std::shared_ptr<UCXXEndpoint> endpoint,
                                                    inflight_requests_t inflight_requests,
                                                    std::shared_ptr<ucxx_request_t> request)
  {
    return std::shared_ptr<UCXXRequest>(new UCXXRequest(endpoint, inflight_requests, request));
  }

  void cancel();

  std::shared_ptr<ucxx_request_t> getHandle();

  ucs_status_t getStatus();

  PyObject* getPyFuture();

  void checkError();

  template <typename Rep, typename Period>
  bool isCompleted(std::chrono::duration<Rep, Period> period);

  bool isCompleted(int64_t periodNs = 0);

  static void callback(void* request, ucs_status_t status, void* arg, std::string operation)
  {
    ucxx_request_t* ucxx_req = (ucxx_request_t*)arg;
    status                   = ucp_request_check_status(request);

    if (ucxx_req == nullptr)
      ucxx_error(
        "error when _callback was called for \"%s\", "
        "probably due to tag_msg() return value being deleted "
        "before completion.",
        operation.c_str());

    ucxx_trace_req("_calback called for \"%s\" with status %d (%s)",
                   operation.c_str(),
                   status,
                   ucs_status_string(status));

    UCXXRequest::setStatus(ucxx_req, ucp_request_check_status(request));

    ucxx_trace_req("ucxx_req->callback: %p", ucxx_req->callback.target<void (*)(void)>());
    if (ucxx_req->callback) ucxx_req->callback(ucxx_req->callback_data);

    ucp_request_free(request);
  }

  static void setStatus(ucxx_request_t* ucxx_req, ucs_status_t status)
  {
    ucxx_req->status = status;

#if UCXX_ENABLE_PYTHON
    auto future = std::static_pointer_cast<PythonFuture>(ucxx_req->py_future);
    future->notify(status);
#endif
  }

  static void process(ucp_worker_h worker,
                      void* request,
                      ucxx_request_t* ucxx_req,
                      std::string operationName)
  {
    ucs_status_t status;

    // Operation completed immediately
    if (request == NULL) {
      status = UCS_OK;
    } else {
      if (UCS_PTR_IS_ERR(request)) {
        status = UCS_PTR_STATUS(request);
      } else if (UCS_PTR_IS_PTR(request)) {
        // Completion will be handled by callback
        ucxx_req->request = request;
        return;
      } else {
        status = UCS_OK;
      }
    }

    UCXXRequest::setStatus(ucxx_req, status);

    ucxx_trace_req("ucxx_req->callback: %p", ucxx_req->callback.target<void (*)(void)>());
    if (ucxx_req->callback) ucxx_req->callback(ucxx_req->callback_data);

    if (status != UCS_OK) {
      ucxx_error("error on %s with status %d (%s)",
                 operationName.c_str(),
                 status,
                 ucs_status_string(status));
      throw UCXXError(std::string("Error on ") + operationName + std::string(" message"));
    } else {
      ucxx_trace_req("%s completed immediately", operationName.c_str());
    }
  }

  // void populateNotificationRequest(std::shared_ptr<NotificationRequest> notificationRequest) = 0;
};

}  // namespace ucxx
