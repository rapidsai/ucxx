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

#ifdef UCXX_ENABLE_PYTHON
#include <Python.h>
#endif

namespace ucxx {

class UCXXRequest : public UCXXComponent {
 private:
  std::shared_ptr<ucxx_request_t> _handle{nullptr};
  inflight_requests_t _inflight_requests{nullptr};

  UCXXRequest(std::shared_ptr<UCXXEndpoint> endpoint,
              inflight_requests_t inflight_requests,
              std::shared_ptr<ucxx_request_t> request)
    : _handle{request}, _inflight_requests{inflight_requests}
  {
    if (endpoint == nullptr || endpoint->getHandle() == nullptr)
      throw ucxx::UCXXError("Endpoint not initialized");

    setParent(endpoint);
  }

 public:
  UCXXRequest() = default;

  UCXXRequest(const UCXXRequest&) = delete;
  UCXXRequest& operator=(UCXXRequest const&) = delete;

  UCXXRequest(UCXXRequest&& o) noexcept
    : _handle{std::exchange(o._handle, nullptr)},
      _inflight_requests{std::exchange(o._inflight_requests, nullptr)}
  {
  }

  UCXXRequest& operator=(UCXXRequest&& o) noexcept
  {
    this->_handle            = std::exchange(o._handle, nullptr);
    this->_inflight_requests = std::exchange(o._inflight_requests, nullptr);

    return *this;
  }

  ~UCXXRequest()
  {
    if (_handle == nullptr) return;

    if (_inflight_requests != nullptr) {
      auto search = _inflight_requests->find(this);
      if (search != _inflight_requests->end()) _inflight_requests->erase(search);
    }
  }

  friend std::shared_ptr<UCXXRequest> createRequest(std::shared_ptr<UCXXEndpoint>& endpoint,
                                                    inflight_requests_t inflight_requests,
                                                    std::shared_ptr<ucxx_request_t> request)
  {
    return std::shared_ptr<UCXXRequest>(new UCXXRequest(endpoint, inflight_requests, request));
  }

  void cancel()
  {
    auto endpoint = std::dynamic_pointer_cast<UCXXEndpoint>(getParent());
    auto worker   = std::dynamic_pointer_cast<UCXXWorker>(endpoint->getParent());
    ucp_request_cancel(worker->get_handle(), _handle->request);
  }

  std::shared_ptr<ucxx_request_t> getHandle() { return _handle; }

  ucs_status_t getStatus() { return _handle->status; }

#ifdef UCXX_ENABLE_PYTHON
  PyObject* getPyFuture() { return (PyObject*)_handle->py_future->getHandle(); }
#endif

  void checkError()
  {
    // Marking the pointer volatile is necessary to ensure the compiler
    // won't optimize the condition out when using a separate worker
    // progress thread
    volatile auto handle = _handle.get();
    switch (handle->status) {
      case UCS_OK:
      case UCS_INPROGRESS: return;
      case UCS_ERR_CANCELED: throw UCXXCanceledError(ucs_status_string(handle->status)); break;
      default: throw UCXXError(ucs_status_string(handle->status)); break;
    }
  }

  template <typename Rep, typename Period>
  bool isCompleted(std::chrono::duration<Rep, Period> period)
  {
    // Marking the pointer volatile is necessary to ensure the compiler
    // won't optimize the condition out when using a separate worker
    // progress thread
    volatile auto handle = _handle.get();
    return handle->status != UCS_INPROGRESS;
  }

  bool isCompleted(int64_t periodNs = 0) { return isCompleted(std::chrono::nanoseconds(periodNs)); }
};

}  // namespace ucxx
