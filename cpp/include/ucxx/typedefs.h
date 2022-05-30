/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <functional>
#include <unordered_map>

namespace ucxx {

class Request;
class PythonFuture;

// Non-blocking requests
typedef struct ucxx_request {
  ucs_status_t status                                 = UCS_INPROGRESS;
  void* request                                       = nullptr;
  std::shared_ptr<PythonFuture> py_future             = nullptr;
  std::function<void(std::shared_ptr<void>)> callback = nullptr;
  std::shared_ptr<void> callback_data                 = nullptr;
} ucxx_request_t;

// Logging levels
typedef enum {
  _LOG_LEVEL_FATAL,       /* Immediate termination */
  _LOG_LEVEL_ERROR,       /* Error is returned to the user */
  _LOG_LEVEL_WARN,        /* Something's wrong, but we continue */
  _LOG_LEVEL_DIAG,        /* Diagnostics, silent adjustments or internal error handling */
  _LOG_LEVEL_INFO,        /* Information */
  _LOG_LEVEL_DEBUG,       /* Low-volume debugging */
  _LOG_LEVEL_TRACE,       /* High-volume debugging */
  _LOG_LEVEL_TRACE_REQ,   /* Every send/receive request */
  _LOG_LEVEL_TRACE_DATA,  /* Data sent/received on the transport */
  _LOG_LEVEL_TRACE_ASYNC, /* Asynchronous progress engine */
  _LOG_LEVEL_TRACE_FUNC,  /* Function calls */
  _LOG_LEVEL_TRACE_POLL,  /* Polling functions */
  _LOG_LEVEL_LAST,
  _LOG_LEVEL_PRINT /* Temporary output */
} ucxx_log_level_t;

typedef std::unordered_map<Request*, std::weak_ptr<Request>> inflight_request_map_t;
typedef std::shared_ptr<inflight_request_map_t> inflight_requests_t;

typedef std::unordered_map<std::string, std::string> ConfigMap;

}  // namespace ucxx
