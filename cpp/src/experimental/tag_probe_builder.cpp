/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>

#include <ucxx/constructors.h>
#include <ucxx/experimental/builder_utils.h>
#include <ucxx/experimental/tag_probe_builder.h>

namespace ucxx::experimental {

struct TagProbeInfoBuilder::Impl {
  bool matched{false};
  ucp_tag_recv_info_t info{};
  ucp_tag_message_h handle{nullptr};

  Impl() = default;

  Impl(const ucp_tag_recv_info_t& tagInfo, ucp_tag_message_h tagHandle)
    : matched(true), info(tagInfo), handle(tagHandle)
  {
  }
};

TagProbeInfoBuilder::TagProbeInfoBuilder() : _impl(std::make_unique<Impl>()) {}

TagProbeInfoBuilder::TagProbeInfoBuilder(const ucp_tag_recv_info_t& info, ucp_tag_message_h handle)
  : _impl(std::make_unique<Impl>(info, handle))
{
}

UCXX_BUILDER_PIMPL_DEFAULTS(TagProbeInfoBuilder, TagProbeInfo)

std::shared_ptr<TagProbeInfo> TagProbeInfoBuilder::build()
{
  if (_impl->matched) return ucxx::createTagProbeInfo(_impl->info, _impl->handle);
  return ucxx::createTagProbeInfo();
}

}  // namespace ucxx::experimental
