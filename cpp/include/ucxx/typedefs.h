/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

namespace ucxx
{

// Non-blocking requests
typedef struct ucxx_request
{
    ucs_status_t status = UCS_INPROGRESS;
} ucxx_request_t;

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
    UCXX_LOG_LEVEL_PRINT        /* Temporary output */
} ucxx_log_level_t;

}  // namespace ucxx
