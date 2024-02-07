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

/**
 * @brief Data for an Active Message send.
 *
 * Type identifying an Active Message send operation and containing data specific to this
 * request type.
 */
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
   * @param[in] buffer      a raw pointer to the data to be sent.
   * @param[in] length      the size in bytes of the message to be sent.
   * @param[in] memoryType  the memory type of the buffer.
   */
  explicit AmSend(const decltype(_buffer) buffer,
                  const decltype(_length) length,
                  const decltype(_memoryType) memoryType = UCS_MEMORY_TYPE_HOST);

  AmSend() = delete;
};

/**
 * @brief Data for an Active Message receive.
 *
 * Type identifying an Active Message receive operation and containing data specific to this
 * request type.
 */
class AmReceive {
 public:
  std::shared_ptr<::ucxx::Buffer> _buffer{nullptr};  ///< The AM received message buffer

  /**
   * @brief Constructor for Active Message-specific receive data.
   *
   * Construct an object containing Active Message-specific receive data. Currently no
   * specific data to receive Active Message is supported, but this class exists to act as
   * an operation identifier, providing interface compatibility.
   */
  AmReceive();
};

/**
 * @brief Data for a memory send.
 *
 * Type identifying a memory send operation and containing data specific to this request type.
 */
class MemSend {
 public:
  const void* _buffer{nullptr};   ///< The raw pointer where data to be sent is stored.
  const size_t _length{0};        ///< The length of the message.
  const uint64_t _remoteAddr{0};  ///< Remote memory address to write to.
  const ucp_rkey_h _rkey{};       ///< UCX remote key associated with the remote memory address.

  /**
   * @brief Constructor for memory-specific data.
   *
   * Construct an object containing memory-specific data.
   *
   * @param[in] buffer      a raw pointer to the data to be sent.
   * @param[in] length      the size in bytes of the tag message to be sent.
   * @param[in] remoteAddr  the destination remote memory address to write to.
   * @param[in] rkey        the remote memory key associated with the remote memory address.
   */
  explicit MemSend(const decltype(_buffer) buffer,
                   const decltype(_length) length,
                   const decltype(_remoteAddr) remoteAddr,
                   const decltype(_rkey) rkey);

  MemSend() = delete;
};

/**
 * @brief Data for a memory receive.
 *
 * Type identifying a memory receive operation and containing data specific to this request
 * type.
 */
class MemReceive {
 public:
  void* _buffer{nullptr};         ///< The raw pointer where received data should be stored.
  const size_t _length{0};        ///< The length of the message.
  const uint64_t _remoteAddr{0};  ///< Remote memory address to read from.
  const ucp_rkey_h _rkey{};       ///< UCX remote key associated with the remote memory address.

  /**
   * @brief Constructor for memory-specific data.
   *
   * Construct an object containing memory-specific data.
   *
   * @param[out] buffer     a raw pointer to the received data.
   * @param[in]  length     the size in bytes of the tag message to be received.
   * @param[in]  remoteAddr the source remote memory address to read from.
   * @param[in]  rkey       the remote memory key associated with the remote memory address.
   */
  explicit MemReceive(decltype(_buffer) buffer,
                      const decltype(_length) length,
                      const decltype(_remoteAddr) remoteAddr,
                      const decltype(_rkey) rkey);

  MemReceive() = delete;
};

/**
 * @brief Data for a Stream send.
 *
 * Type identifying a Stream send operation and containing data specific to this request
 * type.
 */
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

/**
 * @brief Data for an Stream receive.
 *
 * Type identifying an Stream receive operation and containing data specific to this
 * request type.
 */
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

/**
 * @brief Data for a Tag send.
 *
 * Type identifying a Tag send operation and containing data specific to this request type.
 */
class TagSend {
 public:
  const void* _buffer{nullptr};  ///< The raw pointer where data to be sent is stored.
  const size_t _length{0};       ///< The length of the message.
  const ::ucxx::Tag _tag{0};     ///< Tag to match

  /**
   * @brief Constructor for tag-specific data.
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

/**
 * @brief Data for a Tag receive.
 *
 * Type identifying a Tag receive operation and containing data specific to this request
 * type.
 */
class TagReceive {
 public:
  void* _buffer{nullptr};             ///< The raw pointer where received data should be stored.
  const size_t _length{0};            ///< The length of the message.
  const ::ucxx::Tag _tag{0};          ///< Tag to match
  const ::ucxx::TagMask _tagMask{0};  ///< Tag mask to use

  /**
   * @brief Constructor for tag-specific data.
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

/**
 * @brief Data for a multi-buffer Tag send.
 *
 * Type identifying a multi-buffer Tag send operation and containing data specific to this
 * request type.
 */
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
   * @param[in] buffer  a raw pointers to the data to be sent.
   * @param[in] length  the size in bytes of the tag messages to be sent.
   * @param[in] isCUDA  flags indicating whether buffers being sent are CUDA.
   * @param[in] tag     the tags to match.
   */
  explicit TagMultiSend(const decltype(_buffer)& buffer,
                        const decltype(_length)& length,
                        const decltype(_isCUDA)& isCUDA,
                        const decltype(_tag) tag);

  TagMultiSend() = delete;
};

/**
 * @brief Data for a multi-buffer Tag receive.
 *
 * Type identifying a multi-buffer Tag receive operation and containing data specific to
 * this request type.
 */
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
                                 MemSend,
                                 MemReceive,
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

template <class T>
RequestData getRequestData(T t)
{
  return std::visit([](auto arg) -> RequestData { return arg; }, t);
}

}  // namespace data

}  // namespace ucxx
