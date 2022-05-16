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
 private:
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

  friend std::shared_ptr<UCXXRequest> createRequest(std::shared_ptr<UCXXEndpoint>& endpoint,
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
};

}  // namespace ucxx
