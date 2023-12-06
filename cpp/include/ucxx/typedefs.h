/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <ucp/api/ucp.h>

namespace ucxx {

class Buffer;
class Request;

// Logging levels
typedef enum {
  UCXX_LOG_LEVEL_FATAL,       /* Immediate termination */
  UCXX_LOG_LEVEL_ERROR,       /* Error is returned to the user */
  UCXX_LOG_LEVEL_WARN,        /* Something's wrong, but we continue */
  UCXX_LOG_LEVEL_DIAG,        /* Diagnostics, silent adjustments or internal error handling */
  UCXX_LOG_LEVEL_INFO,        /* Information */
  UCXX_LOG_LEVEL_DEBUG,       /* Low-volume debugging */
  UCXX_LOG_LEVEL_TRACE,       /* High-volume debugging */
  UCXX_LOG_LEVEL_TRACE_REQ,   /* Every send/receive request */
  UCXX_LOG_LEVEL_TRACE_DATA,  /* Data sent/received on the transport */
  UCXX_LOG_LEVEL_TRACE_ASYNC, /* Asynchronous progress engine */
  UCXX_LOG_LEVEL_TRACE_FUNC,  /* Function calls */
  UCXX_LOG_LEVEL_TRACE_POLL,  /* Polling functions */
  UCXX_LOG_LEVEL_LAST,
  UCXX_LOG_LEVEL_PRINT /* Temporary output */
} ucxx_log_level_t;

enum class TransferDirection { Undefined = 0, Send, Receive };

enum Tag : ucp_tag_t {};
enum TagMask : ucp_tag_t {};

static constexpr TagMask TagMaskFull{UINT64_MAX};

typedef std::unordered_map<std::string, std::string> ConfigMap;

typedef std::function<void(ucs_status_t, std::shared_ptr<void>)> RequestCallbackUserFunction;
typedef std::shared_ptr<void> RequestCallbackUserData;

typedef std::function<std::shared_ptr<Buffer>(size_t)> AmAllocatorType;

}  // namespace ucxx
