/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <atomic>
#include <memory>
#include <optional>

#include <ucp/api/ucp.h>
#include <ucxx/typedefs.h>

namespace ucxx {

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
 *
 * @note This class is managed via shared_ptr to prevent multiple objects from holding the same
 * handle, which could lead to undefined behavior if the message is received multiple times.
 */
class TagProbeInfo {
 public:
  const bool matched{false};  ///< Whether a message was matched
  const std::optional<TagRecvInfo> info{
    std::nullopt};  ///< Tag receive information (only valid if matched=true)
  const std::optional<ucp_tag_message_h> handle{
    std::nullopt};                            ///< Message handle (only valid if matched=true)
  mutable std::atomic<bool> consumed{false};  ///< Whether the message has been consumed

  TagProbeInfo(const TagProbeInfo&)            = delete;
  TagProbeInfo& operator=(TagProbeInfo const&) = delete;
  TagProbeInfo(TagProbeInfo&& o)               = delete;
  TagProbeInfo& operator=(TagProbeInfo&& o)    = delete;

  ~TagProbeInfo();

  /**
   * @brief Mark the handle as consumed.
   *
   * Call this method after the handle has been used to receive the message,
   * preventing the destructor from issuing a warning about unconsumed handles.
   */
  void consume() const;

  /**
   * @brief Constructor for `shared_ptr<ucxx::TagProbeInfo>`.
   *
   * The constructor for a `shared_ptr<ucxx::TagProbeInfo>` object, initializing
   * `matched` to false and `info` and `handle` as empty optionals.
   *
   * @code{.cpp}
   * auto tagProbeInfo = ucxx::createTagProbeInfo();
   * @endcode
   *
   * @returns The `shared_ptr<ucxx::TagProbeInfo>` object
   */
  friend std::shared_ptr<TagProbeInfo> createTagProbeInfo();

  /**
   * @brief Constructor for `shared_ptr<ucxx::TagProbeInfo>`.
   *
   * The constructor for a `shared_ptr<ucxx::TagProbeInfo>` object, initializing
   * `matched` to true and wrapping the provided `info` and `handle` in optionals.
   *
   * @code{.cpp}
   * auto tagProbeInfo = ucxx::createTagProbeInfo(info, handle);
   * @endcode
   *
   * @param[in] info    The UCP tag receive info structure.
   * @param[in] handle  The UCP tag message handle (can be nullptr if remove=false).
   *
   * @returns The `shared_ptr<ucxx::TagProbeInfo>` object
   */
  friend std::shared_ptr<TagProbeInfo> createTagProbeInfo(const ucp_tag_recv_info_t& info,
                                                          ucp_tag_message_h handle);

 private:
  /**
   * @brief Private constructor of `ucxx::TagProbeInfo`.
   *
   * Initializes `matched` to false and leaves `info` and `handle` as empty optionals.
   *
   * This is the internal implementation of `ucxx::TagProbeInfo` default constructor, made
   * private not to be called directly. This constructor is made private to ensure all UCXX
   * objects * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::createTagProbeInfo()`
   */
  TagProbeInfo() = default;

  /**
   * @brief Private constructor of `ucxx::TagProbeInfo`.
   *
   * Initializes `matched` to true and wraps the provided `info` and `handle` in optionals.
   *
   * This is the internal implementation of `ucxx::TagProbeInfo` default constructor, made
   * private not to be called directly. This constructor is made private to ensure all UCXX
   * objects * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::createTagProbeInfo()`
   *
   * @param[in] info    The UCP tag receive info structure.
   * @param[in] handle  The UCP tag message handle (can be nullptr if remove=false).
   */
  TagProbeInfo(const ucp_tag_recv_info_t& info, ucp_tag_message_h handle);
};

/**
 * @brief Constructor for `shared_ptr<ucxx::TagProbeInfo>`.
 *
 * The constructor for a `shared_ptr<ucxx::TagProbeInfo>` object, initializing
 * `matched` to false and `info` and `handle` as empty optionals.
 *
 * @code{.cpp}
 * auto tagProbeInfo = ucxx::createTagProbeInfo();
 * @endcode
 *
 * @returns The `shared_ptr<ucxx::TagProbeInfo>` object
 */
std::shared_ptr<TagProbeInfo> createTagProbeInfo();

/**
 * @brief Constructor for `shared_ptr<ucxx::TagProbeInfo>`.
 *
 * The constructor for a `shared_ptr<ucxx::TagProbeInfo>` object, initializing
 * `matched` to true and wrapping the provided `info` and `handle` in optionals.
 *
 * @code{.cpp}
 * auto tagProbeInfo = ucxx::createTagProbeInfo(info, handle);
 * @endcode
 *
 * @param[in] info    The UCP tag receive info structure.
 * @param[in] handle  The UCP tag message handle (can be nullptr if remove=false).
 *
 * @returns The `shared_ptr<ucxx::TagProbeInfo>` object
 */
std::shared_ptr<TagProbeInfo> createTagProbeInfo(const ucp_tag_recv_info_t& info,
                                                 ucp_tag_message_h handle);

}  // namespace ucxx
