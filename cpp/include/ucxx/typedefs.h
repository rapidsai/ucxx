/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <optional>
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
 * @brief Information about probed tag message.
 *
 * Contains information returned when probing by a tag message received by the worker but
 * not yet consumed.
 */
class TagRecvInfo {
 public:
  Tag senderTag;  ///< Sender tag
  size_t length;  ///< The size of the received data

  /**
   * @brief Construct a TagRecvInfo object from a UCP tag receive info structure.
   *
   * @param[in] info  The UCP tag receive info structure containing the sender tag and
   *                  received data length.
   */
  explicit TagRecvInfo(const ucp_tag_recv_info_t& info);
};

/**
 * @brief Information about probed tag message.
 *
 * Contains complete information about a probed tag message, including whether a message
 * was matched, the tag receive information, and optionally the message handle for efficient
 * reception when remove=true.
 *
 * @warning Callers must check the `matched` member before accessing `info` or `handle`
 *          to prevent misuse and undefined behavior. When `matched` is false, `info` and
 *          `handle` will be empty (std::nullopt).
 */
class TagProbeInfo {
 public:
  bool matched;                     ///< Whether a message was matched
  std::optional<TagRecvInfo> info;  ///< Tag receive information (valid when matched=true)
  std::optional<ucp_tag_message_h>
    handle;  ///< Message handle for efficient reception (valid when matched=true and remove=true)

  /**
   * @brief Construct a TagProbeInfo object when no message is matched.
   *
   * Initializes `matched` to false and leaves `info` and `handle` as empty optionals.
   */
  TagProbeInfo();

  /**
   * @brief Construct a TagProbeInfo object when a message is matched.
   *
   * @param[in] info    The UCP tag receive info structure.
   * @param[in] handle  The UCP tag message handle (can be nullptr if remove=false).
   *
   * Initializes `matched` to true and wraps the provided `info` and `handle` in optionals.
   */
  TagProbeInfo(const ucp_tag_recv_info_t& info, ucp_tag_message_h handle);
};

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
 * @brief A user-defined function to execute after an endpoint closes.
 *
 * A user-defined function to execute after an endpoint closes, allowing execution of custom
 * code after such event.
 */
typedef RequestCallbackUserFunction EndpointCloseCallbackUserFunction;

/**
 * @brief Data for the user-defined function provided to endpoint close callback.
 *
 * Data passed to the user-defined function provided to the endpoint close callback, which
 * the custom user-defined function may act upon.
 */
typedef RequestCallbackUserData EndpointCloseCallbackUserData;

/**
 * @brief Custom Active Message allocator type.
 *
 * Type for a custom Active Message allocator that can be registered by a user so that the
 * Active Message receiver can allocate a buffer of such type upon receiving message.
 */
typedef std::function<std::shared_ptr<Buffer>(size_t)> AmAllocatorType;

/**
 * @brief Active Message receiver callback.
 *
 * Type for a custom Active Message receiver callback, executed by the remote worker upon
 * Active Message request completion. The first parameter is the request that completed,
 * the second is the handle of the UCX endpoint of the sender.
 */
typedef std::function<void(std::shared_ptr<Request>, ucp_ep_h)> AmReceiverCallbackType;

/**
 * @brief Active Message receiver callback owner name.
 *
 * A string containing the owner's name of an Active Message receiver callback. The owner
 * should be a reasonably unique name, usually identifying the application, to allow other
 * applications to coexist and register their own receiver callbacks.
 */
typedef std::string AmReceiverCallbackOwnerType;

/**
 * @brief Active Message receiver callback identifier.
 *
 * A 64-bit unsigned integer unique identifier type of an Active Message receiver callback.
 */
typedef uint64_t AmReceiverCallbackIdType;

/**
 * @brief Serialized form of Active Message receiver callback information.
 *
 * A string type representing the serialized form of an Active Message receiver callback's
 * information, used for transmission and storage.
 */
typedef const std::string AmReceiverCallbackInfoSerialized;

/**
 * @brief Information of an Active Message receiver callback.
 *
 * Type identifying an Active Message receiver callback's owner name and unique identifier.
 */
class AmReceiverCallbackInfo {
 public:
  const AmReceiverCallbackOwnerType owner;  ///< The owner name of the callback
  const AmReceiverCallbackIdType id;        ///< The unique identifier of the callback

  AmReceiverCallbackInfo() = delete;

  /**
   * @brief Construct an AmReceiverCallbackInfo object.
   *
   * @param[in] owner  The owner name of the callback.
   * @param[in] id     The unique identifier of the callback.
   */
  AmReceiverCallbackInfo(const AmReceiverCallbackOwnerType owner, AmReceiverCallbackIdType id);
};

/**
 * @brief Serialized form of a remote key.
 *
 * A string type representing the serialized form of a remote key, used for transmission
 * and storage of remote memory access information.
 */
typedef const std::string SerializedRemoteKey;

}  // namespace ucxx
