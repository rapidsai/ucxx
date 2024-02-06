/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

#include <ucp/api/ucp.h>

namespace ucxx {

class Buffer;
class Request;
class RequestAm;

/**
 * @brief Available logging levels.
 *
 * Available logging levels that are used to enable specific log types based on user's
 * configuration and also to define appropriate functions to be used in UCXX code to log
 * only when the appropriate level is enabled.
 */
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
  UCXX_LOG_LEVEL_LAST,        /* Last level barrier, not an actual level */
  UCXX_LOG_LEVEL_PRINT        /* Temporary output */
} ucxx_log_level_t;

/**
 * @brief The direction of a UCXX transfer.
 *
 * The direction of a UCXX transfer, can be either `Send` or `Receive`.
 */
enum class TransferDirection { Send = 0, Receive };

/**
 * @brief Strong type for a UCP tag.
 *
 * Strong type for a UCP tag, preventing accidental mixing with wrong types, especially
 * useful to prevent passing an argument in wrong order.
 */
enum Tag : ucp_tag_t {};

/**
 * @brief Strong type for a UCP tag mask.
 *
 * Strong type for a UCP tag mask, preventing accidental mixing with wrong types, especially
 * useful to prevent passing an argument in wrong order.
 */
enum TagMask : ucp_tag_t {};

/**
 * @brief A full UCP tag mask.
 *
 * A convenience constant providing a full UCP tag mask (all bits set).
 */
static constexpr TagMask TagMaskFull{std::numeric_limits<std::underlying_type_t<TagMask>>::max()};

/**
 * @brief A UCP configuration map.
 *
 * A UCP configuration map, with keys being the configuration name and value being the
 * actual value set.
 */
typedef std::unordered_map<std::string, std::string> ConfigMap;

/**
 * @brief A user-defined function to execute as part of a `ucxx::Request` callback.
 *
 * A user-defined function to execute as part of a `ucxx::Request` callback, allowing
 * execution of custom code upon request completion.
 */
typedef std::function<void(ucs_status_t, std::shared_ptr<void>)> RequestCallbackUserFunction;

/**
 * @brief Data for the user-defined function provided to the `ucxx::Request` callback.
 *
 * Data passed to the user-defined function provided to the `ucxx::Request` callback, which
 * the custom user-defined function may act upon.
 */
typedef std::shared_ptr<void> RequestCallbackUserData;

/**
 * @brief Custom Active Message allocator type.
 *
 * Type for a custom Active Message allocator that can be registered by a user so that the
 * Active Message receiver can allocate a buffer of such type upon receiving message.
 */
typedef std::function<std::shared_ptr<Buffer>(size_t)> AmAllocatorType;

/**
 * @brief Active Message receiver callback owner name.
 *
 * A string containing the owner's name of an Active Message receiver callback. The owner
 * should be a reasonably unique name, usually identifying the application, to allow other
 * applications to coexist and register their own receiver callbacks.
 */
typedef std::string_view AmReceiverCallbackOwnerType;

/**
 * @brief Active Message receiver callback identifier.
 *
 * A 64-bit unsigned integer unique identifier type of an Active Message receiver callback.
 */
typedef uint64_t AmReceiverCallbackIdType;

/**
 * @brief Active Message receiver callback.
 *
 * Type for a custom Active Message receiver callback, executed by the remote worker upon
 * Active Message request completion.
 */
typedef std::function<void(std::shared_ptr<Request>)> AmReceiverCallbackType;

}  // namespace ucxx
