/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/tag_probe.h>
#include <ucxx/typedefs.h>

namespace ucxx {

class Buffer;

namespace data {

/**
 * @brief Data for a managed Active Message send.
 *
 * Stores the payload description and managed AM send parameters used by
 * `RequestAmManaged`. The parameters are serialized into UCXX's managed AM header before
 * the send is submitted, allowing the receiver to choose an allocator, preserve the user
 * header, and either match the message with `amManagedRecv()` or route it to a registered
 * receiver callback.
 */
class AmSendManaged {
 public:
  const void* const _buffer{nullptr};      ///< The raw pointer where data to be sent is stored.
  const size_t _length{0};                 ///< Message length in bytes (contiguous datatype only).
  const std::vector<ucp_dt_iov_t> _iov{};  ///< Segments for IOV datatype.
  const size_t _count{0};                  ///< Count passed to `ucp_am_send_nbx`: byte count
                                           ///< for contiguous, number of IOV segments for IOV.
  const uint32_t _flags{UCP_AM_SEND_FLAG_REPLY};              ///< UCP AM send flags.
  const ucp_datatype_t _datatype{ucp_dt_make_contig(1)};      ///< UCP datatype.
  const ucs_memory_type_t _memoryType{UCS_MEMORY_TYPE_HOST};  ///< Memory type used on the operation
  const AmSendMemoryTypePolicy _memoryTypePolicy{
    AmSendMemoryTypePolicy::FallbackToHost};  ///< Receiver allocation policy.
  const std::optional<AmReceiverCallbackInfo> _receiverCallbackInfo{
    std::nullopt};  ///< Owner name and unique identifier of the receiver callback.
  const std::vector<std::byte> _userHeader{};  ///< Opaque user-defined header bytes.

  /**
   * @brief Constructor for managed Active Message send data (contiguous).
   *
   * @param[in] buffer  a raw pointer to the data to be sent.
   * @param[in] length  the size in bytes of the message to be sent.
   * @param[in] params  send parameters controlling datatype/flags/policies.
   */
  explicit AmSendManaged(const decltype(_buffer) buffer,
                         const decltype(_length) length,
                         const AmSendParams& params = AmSendParams{});

  /**
   * @brief Constructor for managed Active Message send data (IOV).
   *
   * @param[in] iov     vector of IOV segments to send.
   * @param[in] params  send parameters controlling datatype/flags/policies.
   */
  explicit AmSendManaged(decltype(_iov) iov, const AmSendParams& params = AmSendParams{});

  AmSendManaged() = delete;
};

/**
 * @brief Data for a managed Active Message receive.
 *
 * Stores the receive-side state populated by the worker's managed AM receive callback. When
 * a managed receive completes, `_buffer` holds the received payload and `_userHeader` holds
 * the sender-provided user header bytes.
 */
class AmReceiveManaged {
 public:
  std::shared_ptr<::ucxx::Buffer> _buffer{nullptr};  ///< The AM received message buffer
  std::vector<std::byte> _userHeader{};              ///< User-defined header bytes from the sender.

  /**
   * @brief Constructor for managed Active Message receive data.
   */
  AmReceiveManaged();
};

/**
 * @brief Data for an endpoint close operation.
 *
 * Type identifying an endpoint close operation and containing data specific to this
 * request type.
 */
class EndpointClose {
 public:
  const bool _force{false};  ///< Whether to force endpoint closing.
  /**
   * @brief Constructor for endpoint close-specific data.
   *
   * @param[in] force   force endpoint close if `true`, flush otherwise.
   */
  explicit EndpointClose(const decltype(_force) force);

  EndpointClose() = delete;
};

/**
 * @brief Data for a flush operation.
 *
 * Type identifying a flush operation and containing data specific to this request type.
 */
class Flush {
 public:
  /**
   * @brief Constructor for flush-specific data.
   */
  Flush();
};

/**
 * @brief Data for a memory send.
 *
 * Type identifying a memory send operation and containing data specific to this request type.
 */
class MemPut {
 public:
  const void* const _buffer{nullptr};  ///< The raw pointer where data to be sent is stored.
  const size_t _length{0};             ///< The length of the message.
  const uint64_t _remoteAddr{0};       ///< Remote memory address to write to.
  const ucp_rkey_h _rkey{};  ///< UCX remote key associated with the remote memory address.

  /**
   * @brief Constructor for memory-specific data.
   *
   * @param[in] buffer      a raw pointer to the data to be sent.
   * @param[in] length      the size in bytes of the message to be sent.
   * @param[in] remoteAddr  the destination remote memory address to write to.
   * @param[in] rkey        the remote memory key associated with the remote memory address.
   */
  explicit MemPut(const decltype(_buffer) buffer,
                  const decltype(_length) length,
                  const decltype(_remoteAddr) remoteAddr,
                  const decltype(_rkey) rkey);

  MemPut() = delete;
};

/**
 * @brief Data for a memory receive.
 *
 * Type identifying a memory receive operation and containing data specific to this request type.
 */
class MemGet {
 public:
  void* _buffer{nullptr};         ///< The raw pointer where received data should be stored.
  const size_t _length{0};        ///< The length of the message.
  const uint64_t _remoteAddr{0};  ///< Remote memory address to read from.
  const ucp_rkey_h _rkey{};       ///< UCX remote key associated with the remote memory address.

  /**
   * @brief Constructor for memory-specific data.
   *
   * @param[out] buffer     a raw pointer to the received data.
   * @param[in]  length     the size in bytes of the message to be received.
   * @param[in]  remoteAddr the source remote memory address to read from.
   * @param[in]  rkey       the remote memory key associated with the remote memory address.
   */
  explicit MemGet(decltype(_buffer) buffer,
                  const decltype(_length) length,
                  const decltype(_remoteAddr) remoteAddr,
                  const decltype(_rkey) rkey);

  MemGet() = delete;
};

/**
 * @brief Data for a Stream send.
 *
 * Type identifying a Stream send operation and containing data specific to this request type.
 */
class StreamSend {
 public:
  const void* const _buffer{nullptr};  ///< The raw pointer where data to be sent is stored.
  const size_t _length{0};             ///< The length of the message.

  /**
   * @brief Constructor for stream-specific data.
   *
   * @param[in] buffer  a raw pointer to the data to be sent.
   * @param[in] length  the size in bytes of the message to be sent.
   */
  explicit StreamSend(const decltype(_buffer) buffer, const decltype(_length) length);

  StreamSend() = delete;
};

/**
 * @brief Data for a Stream receive.
 *
 * Type identifying a Stream receive operation and containing data specific to this request type.
 */
class StreamReceive {
 public:
  void* _buffer{nullptr};     ///< The raw pointer where received data should be stored.
  const size_t _length{0};    ///< The expected message length.
  size_t _lengthReceived{0};  ///< The actual received message length.

  /**
   * @brief Constructor for stream-specific data.
   *
   * @param[out] buffer   a raw pointer to the received data.
   * @param[in]  length   the size in bytes of the message to be received.
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
  const void* const _buffer{nullptr};  ///< The raw pointer where data to be sent is stored.
  const size_t _length{0};             ///< The length of the message.
  const ::ucxx::Tag _tag{0};           ///< Tag to match

  /**
   * @brief Constructor for tag-specific data.
   *
   * @param[in] buffer  a raw pointer to the data to be sent.
   * @param[in] length  the size in bytes of the message to be sent.
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
 * Type identifying a Tag receive operation and containing data specific to this request type.
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
   * @param[out] buffer   a raw pointer to the received data.
   * @param[in]  length   the size in bytes of the message to be received.
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
 * @brief Data for a Tag receive using a message handle.
 *
 * Type identifying a Tag receive operation using a message handle and containing data
 * specific to this request type.
 */
class TagReceiveWithHandle {
 public:
  void* _buffer{nullptr};  ///< The raw pointer where received data should be stored.
  const std::shared_ptr<TagProbeInfo> _probeInfo{
    nullptr};  ///< TagProbeInfo containing message length and handle

  /**
   * @brief Constructor for tag receive with handle-specific data.
   *
   * @param[out] buffer      a raw pointer to the received data.
   * @param[in]  probeInfo   the TagProbeInfo object containing message length and handle.
   */
  explicit TagReceiveWithHandle(decltype(_buffer) buffer, std::shared_ptr<TagProbeInfo> probeInfo);

  TagReceiveWithHandle() = delete;
};

/**
 * @brief Data for a multi-buffer Tag send.
 *
 * Type identifying a multi-buffer Tag send operation and containing data specific to this
 * request type.
 */
class TagMultiSend {
 public:
  const std::vector<const void*> _buffer{};  ///< Raw pointers where data to be sent is stored.
  const std::vector<size_t> _length{};       ///< Lengths of messages.
  const std::vector<int> _isCUDA{};  ///< Flags indicating whether the buffer is CUDA or not.
  const ::ucxx::Tag _tag{0};         ///< Tag to match

  /**
   * @brief Constructor for send multi-buffer tag-specific data.
   *
   * @param[in] buffer  raw pointers to the data to be sent.
   * @param[in] length  sizes in bytes of the messages to be sent.
   * @param[in] isCUDA  flags indicating whether buffers being sent are CUDA.
   * @param[in] tag     the tag to match.
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
   * @param[in]  tag      the tag to match.
   * @param[in]  tagMask  the tag mask to use (only used for receive operations).
   */
  explicit TagMultiReceive(const decltype(_tag) tag, const decltype(_tagMask) tagMask);

  TagMultiReceive() = delete;
};

using RequestData = std::variant<std::monostate,
                                 AmSendManaged,
                                 AmReceiveManaged,
                                 EndpointClose,
                                 Flush,
                                 MemPut,
                                 MemGet,
                                 StreamSend,
                                 StreamReceive,
                                 TagSend,
                                 TagReceive,
                                 TagReceiveWithHandle,
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
