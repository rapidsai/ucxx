/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <variant>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/typedefs.h>

namespace ucxx {

class Buffer;

namespace data {

class AmSend {
 public:
  const void* _buffer{nullptr};  ///< The raw pointer where data to be sent is stored.
  const size_t _length{0};       ///< The length of the message.
  const ucs_memory_type_t _memoryType{UCS_MEMORY_TYPE_HOST};  ///< Memory type used on the operation

  /**
   * @brief Constructor for Active Message-specific send data.
   *
   * Construct an object containing Active Message-specific send data.
   *
   * @param[in] memoryType  the memory type of the buffer.
   */
  explicit AmSend(const decltype(_buffer) buffer,
                  const decltype(_length) length,
                  const decltype(_memoryType) memoryType = UCS_MEMORY_TYPE_HOST);

  AmSend() = delete;
};

class AmReceive {
 public:
  std::shared_ptr<::ucxx::Buffer> _buffer{nullptr};  ///< The AM received message buffer

  /**
   * @brief Constructor for Active Message-specific receive data.
   *
   * Construct an object containing Active Message-specific receive data.
   *
   * @param[in] memoryType  the memory type of the buffer.
   */
  AmReceive();
};

class StreamSend {
 public:
  const void* _buffer{nullptr};  ///< The raw pointer where data to be sent is stored.
  const size_t _length{0};       ///< The length of the message.

  /**
   * @brief Constructor for stream-specific data.
   *
   * Construct an object containing stream-specific data.
   *
   * @param[in] buffer  a raw pointer to the data to be sent.
   * @param[in] length  the size in bytes of the tag message to be sent.
   */
  explicit StreamSend(const decltype(_buffer) buffer, const decltype(_length) length);

  StreamSend() = delete;
};

class StreamReceive {
 public:
  void* _buffer{nullptr};     ///< The raw pointer where received data should be stored.
  const size_t _length{0};    ///< The expected messaged length.
  size_t _lengthReceived{0};  ///< The actual received message length.

  /**
   * @brief Constructor for stream-specific data.
   *
   * Construct an object containing stream-specific data.
   *
   * @param[out] buffer   a raw pointer to the received data.
   * @param[in]  length   the size in bytes of the tag message to be received.
   */
  explicit StreamReceive(decltype(_buffer) buffer, const decltype(_length) length);

  StreamReceive() = delete;
};

class TagSend {
 public:
  const void* _buffer{nullptr};  ///< The raw pointer where data to be sent is stored.
  const size_t _length{0};       ///< The length of the message.
  const ::ucxx::Tag _tag{0};     ///< Tag to match

  /**
   * @brief Constructor for tag/multi-buffer tag-specific data.
   *
   * Construct an object containing tag-specific data.
   *
   * @param[in] buffer  a raw pointer to the data to be sent.
   * @param[in] length  the size in bytes of the tag message to be sent.
   * @param[in] tag     the tag to match.
   */
  explicit TagSend(const decltype(_buffer) buffer,
                   const decltype(_length) length,
                   const decltype(_tag) tag);

  TagSend() = delete;
};

class TagReceive {
 public:
  void* _buffer{nullptr};             ///< The raw pointer where received data should be stored.
  const size_t _length{0};            ///< The length of the message.
  const ::ucxx::Tag _tag{0};          ///< Tag to match
  const ::ucxx::TagMask _tagMask{0};  ///< Tag mask to use

  /**
   * @brief Constructor send tag-specific data.
   *
   * Construct an object containing send tag-specific data.
   *
   * @param[out] buffer   a raw pointer to the received data.
   * @param[in]  length   the size in bytes of the tag message to be received.
   * @param[in]  tag      the tag to match.
   * @param[in]  tagMask  the tag mask to use (only used for receive operations).
   */
  explicit TagReceive(decltype(_buffer) buffer,
                      const decltype(_length) length,
                      const decltype(_tag) tag,
                      const decltype(_tagMask) tagMask);

  TagReceive() = delete;
};

class TagMultiSend {
 public:
  const std::vector<void*> _buffer{};   ///< Raw pointers where data to be sent is stored.
  const std::vector<size_t> _length{};  ///< Lengths of messages.
  const std::vector<int> _isCUDA{};     ///< Flags indicating whether the buffer is CUDA or not.
  const ::ucxx::Tag _tag{0};            ///< Tag to match

  /**
   * @brief Constructor for send multi-buffer tag-specific data.
   *
   * Construct an object containing tag/multi-buffer tag-specific data.
   *
   * @param[in] buffer  a raw pointer to the data to be sent.
   * @param[in] length  the size in bytes of the tag message to be sent.
   * @param[in] tag     the tag to match.
   */
  explicit TagMultiSend(const decltype(_buffer)& buffer,
                        const decltype(_length)& length,
                        const decltype(_isCUDA)& isCUDA,
                        const decltype(_tag) tag);

  TagMultiSend() = delete;
};

class TagMultiReceive {
 public:
  const ::ucxx::Tag _tag{0};          ///< Tag to match
  const ::ucxx::TagMask _tagMask{0};  ///< Tag mask to use

  /**
   * @brief Constructor for receive multi-buffer tag-specific data.
   *
   * Construct an object containing receive multi-buffer tag-specific data.
   *
   * @param[in]  tag      the tag to match.
   * @param[in]  tagMask  the tag mask to use (only used for receive operations).
   */
  explicit TagMultiReceive(const decltype(_tag) tag, const decltype(_tagMask) tagMask);

  TagMultiReceive() = delete;
};

using RequestData = std::variant<std::monostate,
                                 AmSend,
                                 AmReceive,
                                 StreamSend,
                                 StreamReceive,
                                 TagSend,
                                 TagReceive,
                                 TagMultiSend,
                                 TagMultiReceive>;

template <class... Ts>
struct dispatch : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
dispatch(Ts...) -> dispatch<Ts...>;
template <class... Ts>
dispatch(Ts&...) -> dispatch<Ts...>;

}  // namespace data

}  // namespace ucxx
