/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
 * @brief Active Message data information for delayed receiving.
 *
 * Structure containing the Active Message data pointer and length that will be used
 * when the user chooses to delay receiving and handle ucp_am_recv_data_nbx manually.
 */
struct AmData {
  void* data;     ///< The Active Message data pointer from the receive callback
  size_t length;  ///< The length of the Active Message data

  AmData() : data(nullptr), length(0) {}

  /**
   * @brief Construct an AmData object.
   *
   * @param[in] data    The Active Message data pointer from the receive callback.
   * @param[in] length  The length of the Active Message data.
   */
  AmData(void* data, size_t length) : data(data), length(length) {}
};

/**
 * @brief Container for arbitrary user header data that can be attached to Active Messages.
 *
 * This class provides a type-safe interface for storing arbitrary user header data that will be
 * serialized and transmitted with Active Messages. It supports common data types while
 * also allowing direct access to the underlying byte storage for custom serialization.
 */
class AmUserHeader {
 private:
  std::vector<uint8_t> _data;

 public:
  AmUserHeader() = delete;

  /**
   * @brief Construct AmUserHeader from a byte array.
   *
   * @param[in] data Pointer to the data to copy.
   * @param[in] size Size of the data in bytes.
   *
   * @throws std::invalid_argument if data is null or if size is 0.
   */
  AmUserHeader(const void* data, size_t size)
  {
    if (size == 0) { throw std::invalid_argument("AmUserHeader size must be greater than zero"); }
    if (data == nullptr) {
      throw std::invalid_argument("AmUserHeader data pointer cannot be null");
    }
    _data.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
  }

  /**
   * @brief Construct AmUserHeader from a string.
   *
   * @param[in] str The string to store.
   *
   * @throws std::invalid_argument if the string is empty.
   */
  explicit AmUserHeader(const std::string& str) : _data(str.begin(), str.end())
  {
    if (str.empty()) { throw std::invalid_argument("AmUserHeader string cannot be empty"); }
  }

  /**
   * @brief Construct AmUserHeader from a vector of bytes.
   *
   * @param[in] data The byte vector to copy.
   *
   * @throws std::invalid_argument if the vector is empty.
   */
  explicit AmUserHeader(const std::vector<uint8_t>& data) : _data(data)
  {
    if (data.empty()) { throw std::invalid_argument("AmUserHeader vector cannot be empty"); }
  }

  /**
   * @brief Construct AmUserHeader from a vector of bytes (move constructor).
   *
   * @param[in] data The byte vector to move.
   *
   * @throws std::invalid_argument if the vector is empty.
   */
  explicit AmUserHeader(std::vector<uint8_t>&& data) : _data(std::move(data))
  {
    if (_data.empty()) { throw std::invalid_argument("AmUserHeader vector cannot be empty"); }
  }

  /**
   * @brief Template constructor for POD types.
   *
   * @param[in] value The POD value to store.
   */
  template <typename T>
  explicit AmUserHeader(const T& value)
    : _data(reinterpret_cast<const uint8_t*>(&value),
            reinterpret_cast<const uint8_t*>(&value) + sizeof(T))
  {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
    static_assert(sizeof(T) > 0, "Type size must be greater than zero");
  }

  /**
   * @brief Get the underlying data as a byte array.
   *
   * @returns Pointer to the underlying data.
   */
  [[nodiscard]] const uint8_t* data() const { return _data.data(); }

  /**
   * @brief Get the size of the data in bytes.
   *
   * @returns Size of the data in bytes.
   */
  [[nodiscard]] size_t size() const { return _data.size(); }

  /**
   * @brief Check if the user data is empty.
   *
   * @returns True if no data is stored, false otherwise.
   */
  [[nodiscard]] bool empty() const { return _data.empty(); }

  /**
   * @brief Get the data as a string.
   *
   * @returns String representation of the data.
   */
  [[nodiscard]] std::string asString() const { return std::string(_data.begin(), _data.end()); }

  /**
   * @brief Get the data as a specific POD type.
   *
   * @returns Reference to the data interpreted as type T.
   * @throws std::runtime_error if the size doesn't match.
   */
  template <typename T>
  [[nodiscard]] const T& as() const
  {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
    if (_data.size() != sizeof(T)) {
      throw std::runtime_error("AmUserHeader size mismatch: expected " + std::to_string(sizeof(T)) +
                               " bytes, got " + std::to_string(_data.size()));
    }
    return *reinterpret_cast<const T*>(_data.data());
  }

  /**
   * @brief Get a copy of the underlying byte vector.
   *
   * @returns Copy of the underlying data vector.
   */
  [[nodiscard]] std::vector<uint8_t> getBytes() const { return _data; }
};

/**
 * @brief Information of an Active Message receiver callback.
 *
 * Type identifying an Active Message receiver callback's owner name and unique identifier.
 */
class AmReceiverCallbackInfo {
 public:
  const AmReceiverCallbackOwnerType owner;  ///< The owner name of the callback
  const AmReceiverCallbackIdType id;        ///< The unique identifier of the callback
  const bool delayReceive;                  ///< Whether to delay receiving data (user-controlled)
  const std::optional<AmUserHeader> userHeader;  ///< Optional arbitrary user header data

  AmReceiverCallbackInfo() = delete;

  /**
   * @brief Construct an AmReceiverCallbackInfo object.
   *
   * @param[in] owner         The owner name of the callback.
   * @param[in] id            The unique identifier of the callback.
   * @param[in] delayReceive  Whether to delay receiving data, allowing user to control when
   * ucp_am_recv_data_nbx is called.
   * @param[in] userHeader    Optional arbitrary user header data to be transmitted with the AM.
   */
  AmReceiverCallbackInfo(const AmReceiverCallbackOwnerType owner,
                         AmReceiverCallbackIdType id,
                         bool delayReceive                      = false,
                         std::optional<AmUserHeader> userHeader = std::nullopt);
};

/**
 * @brief Active Message receiver callback.
 *
 * Type for a custom Active Message receiver callback, executed by the remote worker upon
 * Active Message request completion. The first parameter is the request that completed,
 * the second is the handle of the UCX endpoint of the sender.
 */
typedef std::function<void(std::shared_ptr<Request>, ucp_ep_h, AmReceiverCallbackInfo&)>
  AmReceiverCallbackType;

/**
 * @brief Serialized form of a remote key.
 *
 * A string type representing the serialized form of a remote key, used for transmission
 * and storage of remote memory access information.
 */
typedef const std::string SerializedRemoteKey;

}  // namespace ucxx
