/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>

#include <ucxx/log.h>
#include <ucxx/tag_probe.h>

namespace ucxx {

TagRecvInfo::TagRecvInfo(const ucp_tag_recv_info_t& info)
  : senderTag(Tag(info.sender_tag)), length(info.length)
{
}

TagProbeInfo::TagProbeInfo(const ucp_tag_recv_info_t& info, ucp_tag_message_h handle)
  : matched(true),
    info(TagRecvInfo(info)),
    handle(handle != nullptr ? std::optional<ucp_tag_message_h>(handle) : std::nullopt),
    consumed(false)
{
}

std::shared_ptr<TagProbeInfo> createTagProbeInfo()
{
  return std::shared_ptr<TagProbeInfo>(new TagProbeInfo());
}

std::shared_ptr<TagProbeInfo> createTagProbeInfo(const ucp_tag_recv_info_t& info,
                                                 ucp_tag_message_h handle)
{
  return std::shared_ptr<TagProbeInfo>(new TagProbeInfo(info, handle));
}

TagProbeInfo::~TagProbeInfo()
{
  // Check if handle is populated and has not been consumed
  if (matched && handle.has_value() && handle.value() != nullptr && !consumed) {
    ucxx_warn(
      "ucxx::TagProbeInfo::%s, destroying %p unconsumed message handle %p detected from tag 0x%lx "
      "with "
      "length %lu. ucxx::Worker::tagRecvWithHandle() must be called to consume the handle.",
      __func__,
      this,
      handle.value(),
      info.value().senderTag,
      info.value().length);
  }
}

void TagProbeInfo::consume() const { consumed = true; }

}  // namespace ucxx
