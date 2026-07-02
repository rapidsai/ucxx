/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include <ucp/api/ucp.h>

namespace ucxx {

class AmMessage;
class Buffer;
class Request;
class RequestAmManaged;

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
 * @brief Active Message handler ID type.
 *
 * Strong type alias for the AM handler ID passed to `ucp_worker_set_am_recv_handler`
 * and `ucp_am_send_nbx`.
 */
typedef uint16_t AmHandlerId;

/**
 * @brief Active Message handler callback type.
 *
 * Type for a callback registered via `ucxx::Worker::setAmHandler`. The handler receives
 * an `AmMessage` reference that exposes the header, payload, and message kind
 * (eager vs. rendezvous). For rendezvous messages, the handler must call
 * `AmMessage::receive()` before returning. The handler returns `void`; UCXX infers the
 * UCX status from the handler's actions:
 *
 * `receive()` called: `UCS_INPROGRESS`.
 * `reject(reason)` called: `reason`.
 * Neither action on the eager path: `UCS_OK`.
 *
 * The handler must not allow exceptions to escape: the callback is invoked from a C
 * function pointer registered with UCX, and exceptions propagating through C frames
 * produce undefined behaviour. Any escaping exception is caught by the AM receive callback,
 * logged as a warning, and mapped to `UCS_ERR_IO_ERROR`.
 */
typedef std::function<void(AmMessage&)> AmHandlerType;

/**
 * @brief Contiguous buffer payload for `Endpoint::amSend`.
 *
 * Aggregate type: construct with brace-initialization, e.g.
 * `AmSendContig{buf, size}` or `AmSendContig{buf, size, myDatatype}`.
 */
struct AmSendContig {
  const void* buffer{nullptr};                     ///< Data buffer to send.
  size_t count{0};                                 ///< Number of bytes to send.
  ucp_datatype_t datatype{ucp_dt_make_contig(1)};  ///< UCP datatype.
};

/**
 * @brief Scatter-gather IOV payload for `Endpoint::amSend`.
 *
 * Aggregate type: construct with brace-initialization, e.g.
 * `AmSendIov{std::move(iovVector)}`.
 */
struct AmSendIov {
  std::vector<ucp_dt_iov_t> iov;  ///< Scatter-gather segment list.
};

/**
 * @brief Payload variant for `Endpoint::amSend`.
 *
 * Holds either a contiguous buffer (`AmSendContig`) or a scatter-gather
 * IOV list (`AmSendIov`). The endpoint method dispatches on the active
 * alternative to build the underlying `ucp_am_send_nbx` call.
 */
using AmSendBuffer = std::variant<AmSendContig, AmSendIov>;

/**
 * @brief Custom Active Message allocator type.
 *
 * Type for a custom Active Message allocator that can be registered by a user so that the
 * Active Message receiver can allocate a buffer of such type upon receiving message.
 * Used by the managed AM API.
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
 * @brief Maximum number of usable characters in an Active Message receiver callback owner name.
 */
static constexpr size_t AmReceiverCallbackOwnerMaxLen = 63;

/**
 * @brief On-wire and in-memory storage size for an Active Message receiver callback owner name.
 */
static constexpr size_t AmReceiverCallbackOwnerStorageSize = AmReceiverCallbackOwnerMaxLen + 1;

/**
 * @brief Active Message receiver callback owner name (fixed-size).
 *
 * A fixed-size identifier for the owner of an Active Message receiver callback. The owner
 * should be a reasonably unique name, usually identifying the application, to allow other
 * applications to coexist and register their own receiver callbacks.
 *
 * Names are stored in a fixed 64-byte buffer, zero-padded. The maximum string length is
 * 63 characters. Attempting to construct from a longer string throws @c std::invalid_argument.
 *
 * Implicit construction from `const char*` and `std::string` is supported so that existing
 * call sites such as `AmReceiverCallbackInfo("MyApp", 0)` compile unchanged.
 */
class AmReceiverCallbackOwnerType {
 public:
  /** @brief Construct an empty (all-zero) owner name. */
  AmReceiverCallbackOwnerType() = default;

  /** @brief Construct from a null-terminated C string. Throws if length exceeds the limit. */
  AmReceiverCallbackOwnerType(const char* s)  // NOLINT(runtime/explicit)
  {
    if (s == nullptr) return;
    const size_t len = std::strlen(s);
    if (len > AmReceiverCallbackOwnerMaxLen)
      throw std::invalid_argument(
        "AmReceiverCallbackOwnerType: owner name exceeds "
        "maximum length of " +
        std::to_string(AmReceiverCallbackOwnerMaxLen) + " characters");
    std::memcpy(_data.data(), s, len);
  }

  /** @brief Construct from a @c std::string. Throws if length exceeds the limit. */
  AmReceiverCallbackOwnerType(const std::string& s)  // NOLINT(runtime/explicit)
    : AmReceiverCallbackOwnerType(s.c_str())
  {
  }

  /** @brief Pointer to the raw fixed-size storage. */
  [[nodiscard]] const char* data() const noexcept { return _data.data(); }

  /** @brief Mutable pointer to the raw fixed-size storage (for deserialization). */
  [[nodiscard]] char* data() noexcept { return _data.data(); }

  /** @brief The fixed storage size that is always sent on the wire. */
  static constexpr size_t storageSize() noexcept { return AmReceiverCallbackOwnerStorageSize; }

  /** @brief Equality comparison. */
  bool operator==(const AmReceiverCallbackOwnerType& other) const noexcept
  {
    return _data == other._data;
  }

  /** @brief Inequality comparison. */
  bool operator!=(const AmReceiverCallbackOwnerType& other) const noexcept
  {
    return !(*this == other);
  }

 private:
  std::array<char, AmReceiverCallbackOwnerStorageSize> _data{};
};

/**
 * @brief Active Message receiver callback identifier.
 *
 * A 64-bit unsigned integer unique identifier type of an Active Message receiver callback.
 */
typedef uint64_t AmReceiverCallbackIdType;

/**
 * @brief Information of an Active Message receiver callback.
 *
 * Type identifying an Active Message receiver callback's owner name and unique identifier.
 */
class AmReceiverCallbackInfo {
 public:
  AmReceiverCallbackOwnerType owner;  ///< The owner name of the callback
  AmReceiverCallbackIdType id;        ///< The unique identifier of the callback

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
 * @brief Policy used to allocate receive buffers for Active Messages.
 *
 * Active Message receive allocations can be strict (error if no allocator is registered for
 * sender-provided memory type) or permissive (fallback to host allocation).
 */
enum class AmSendMemoryTypePolicy {
  FallbackToHost = 0,  ///< If no allocator exists for memory type, fallback to host memory.
  ErrorOnUnsupported,  ///< If no allocator exists for memory type, fail with unsupported error.
};

/**
 * @brief Parameters controlling Active Message send behavior.
 *
 * This object is used by the extended Active Message API to expose UCX send knobs without
 * breaking existing callers.
 */
struct AmSendParams {
  uint32_t flags{UCP_AM_SEND_FLAG_REPLY};              ///< UCP AM send flags.
  ucp_datatype_t datatype{ucp_dt_make_contig(1)};      ///< Datatype used by `ucp_am_send_nbx`.
  ucs_memory_type_t memoryType{UCS_MEMORY_TYPE_HOST};  ///< Sender memory type hint.
  AmSendMemoryTypePolicy memoryTypePolicy{
    AmSendMemoryTypePolicy::FallbackToHost};  ///< Receiver allocation policy.
  std::optional<AmReceiverCallbackInfo> receiverCallbackInfo{
    std::nullopt};                      ///< Optional receiver callback metadata.
  std::vector<std::byte> userHeader{};  ///< Opaque user-defined header bytes. This is serialized
                                        ///< into the AM header parameter of `ucp_am_send_nbx`,
                                        ///< which is subject to transport-level size limits. For
                                        ///< TCP, the default segment size is ~8 KiB
                                        ///< (`UCX_TCP_TX_SEG_SIZE` / `UCX_TCP_RX_SEG_SIZE`).
                                        ///< Headers that exceed the transport limit will cause a
                                        ///< fatal UCX error. Keep user headers small
                                        ///< (recommended < 4 KiB) or increase segment size env
                                        ///< vars as needed.

  /**
   * @brief Set opaque user header bytes from raw pointer.
   *
   * @param[in] data  pointer to input bytes, may be `nullptr` iff `size == 0`.
   * @param[in] size  number of bytes in input.
   */
  void setUserHeader(const void* data, size_t size)
  {
    if (size > 0 && data == nullptr)
      throw std::invalid_argument(
        "AmSendParams::setUserHeader received null data with non-zero size");
    userHeader.resize(size);
    if (size > 0) memcpy(userHeader.data(), data, size);
  }

  /**
   * @brief Convenience overload to set user header from string-like views.
   *
   * @param[in] data view of opaque bytes.
   */
  void setUserHeader(std::string_view data) { setUserHeader(data.data(), data.size()); }
};

/**
 * @brief Serialized form of a remote key.
 *
 * A string type representing the serialized form of a remote key, used for transmission
 * and storage of remote memory access information.
 */
typedef const std::string SerializedRemoteKey;

/**
 * @brief Hash functor for @c AmReceiverCallbackOwnerType.
 *
 * Hashes the full fixed-size storage so that zero-padded names compare deterministically.
 * Used as the hasher for @c std::unordered_map keyed by @c AmReceiverCallbackOwnerType.
 */
struct AmReceiverCallbackOwnerHash {
  /** @brief Compute hash of an @c AmReceiverCallbackOwnerType. */
  size_t operator()(const AmReceiverCallbackOwnerType& o) const noexcept
  {
    return std::hash<std::string_view>{}(
      std::string_view(o.data(), AmReceiverCallbackOwnerStorageSize));
  }
};

}  // namespace ucxx
