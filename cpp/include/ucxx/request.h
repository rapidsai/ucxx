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

  UCXXRequest(std::shared_ptr<UCXXEndpoint> endpoint,
              std::shared_ptr<ucxx_request_t> request,
              std::shared_ptr<NotificationRequest> notificationRequest);

  void process(ucp_worker_h worker, void* request, std::string operationName);

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

  void setStatus(ucs_status_t status)
  {
    _handle->status = status;

#if UCXX_ENABLE_PYTHON
    auto future = std::static_pointer_cast<PythonFuture>(_handle->py_future);
    future->notify(status);
#endif
  }

  static std::shared_ptr<ucxx_request_t> createRequestBase(
    std::shared_ptr<UCXXWorker> worker,
    std::function<void(std::shared_ptr<void>)> callbackFunction = nullptr,
    std::shared_ptr<void> callbackData                          = nullptr)
  {
    auto request = std::make_shared<ucxx_request_t>();
#if UCXX_ENABLE_PYTHON
    request->py_future = worker->getPythonFuture();
    ucxx_trace_req("request->py_future: %p", request->py_future.get());
#endif
    request->callback      = callbackFunction;
    request->callback_data = callbackData;

    return request;
  }

  virtual void populateNotificationRequest() = 0;
};

}  // namespace ucxx
