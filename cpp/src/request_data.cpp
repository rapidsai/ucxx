/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <stdexcept>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucp/api/ucp_def.h>
#include <ucxx/request_data.h>
#include <ucxx/typedefs.h>

namespace ucxx {

namespace data {

AmSend::AmSend(const void* buffer,
               const size_t length,
               const ucs_memory_type memoryType,
               const std::optional<AmReceiverCallbackInfo> receiverCallbackInfo)
  : _buffer(buffer),
    _length(length),
    _memoryType(memoryType),
    _receiverCallbackInfo(receiverCallbackInfo)
{
}

AmReceive::AmReceive() {}

EndpointClose::EndpointClose(const bool force) : _force(force) {}

Flush::Flush() {}

MemPut::MemPut(const void* buffer,
               const size_t length,
               const uint64_t remoteAddr,
               const ucp_rkey_h rkey)
  : _buffer(buffer), _length(length), _remoteAddr(remoteAddr), _rkey(rkey)
{
}

MemGet::MemGet(void* buffer, const size_t length, const uint64_t remoteAddr, const ucp_rkey_h rkey)
  : _buffer(buffer), _length(length), _remoteAddr(remoteAddr), _rkey(rkey)
{
}

StreamSend::StreamSend(const void* buffer, const size_t length) : _buffer(buffer), _length(length)
{
  /**
   * Stream API does not support zero-sized messages. See
   * https://github.com/openucx/ucx/blob/6b45097e32c75c9b5d17f4770923204d568548d0/src/ucp/stream/stream_recv.c#L501
   */
  if (buffer == nullptr) throw std::runtime_error("Buffer cannot be a nullptr.");
  if (length == 0) throw std::runtime_error("Length has to be a positive value.");
}

StreamReceive::StreamReceive(void* buffer, const size_t length) : _buffer(buffer), _length(length)
{
  /**
   * Stream API does not support zero-sized messages. See
   * https://github.com/openucx/ucx/blob/6b45097e32c75c9b5d17f4770923204d568548d0/src/ucp/stream/stream_recv.c#L501
   */
  if (buffer == nullptr) throw std::runtime_error("Buffer cannot be a nullptr.");
  if (length == 0) throw std::runtime_error("Length has to be a positive value.");
}

TagSend::TagSend(const void* buffer, const size_t length, const ::ucxx::Tag tag)
  : _buffer(buffer), _length(length), _tag(tag)
{
}

TagReceive::TagReceive(void* buffer,
                       const size_t length,
                       const ::ucxx::Tag tag,
                       const ::ucxx::TagMask tagMask)
  : _buffer(buffer), _length(length), _tag(tag), _tagMask(tagMask)
{
}

TagMultiSend::TagMultiSend(const std::vector<void*>& buffer,
                           const std::vector<size_t>& length,
                           const std::vector<int>& isCUDA,
                           const ::ucxx::Tag tag)
  : _buffer(buffer), _length(length), _isCUDA(isCUDA), _tag(tag)
{
  if (length.size() != buffer.size() || isCUDA.size() != buffer.size())
    throw std::runtime_error("All input vectors should be of equal size");
}

TagMultiReceive::TagMultiReceive(const ::ucxx::Tag tag, const ::ucxx::TagMask tagMask)
  : _tag(tag), _tagMask(tagMask)
{
}

}  // namespace data

}  // namespace ucxx
