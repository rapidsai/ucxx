/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <ucxx/context.h>
#include <ucxx/experimental/builder_utils.h>
#include <ucxx/experimental/context_builder.h>

namespace ucxx {

namespace experimental {

struct ContextBuilder::Impl {
  ConfigMap configMap{};
  uint64_t featureFlags;
};

ContextBuilder::ContextBuilder(uint64_t featureFlags) : _impl(std::make_unique<Impl>())
{
  _impl->featureFlags = featureFlags;
}

UCXX_BUILDER_PIMPL_DEFAULTS(ContextBuilder, Context)

ContextBuilder& ContextBuilder::configMap(ConfigMap configMap)
{
  _impl->configMap = std::move(configMap);
  return *this;
}

std::shared_ptr<Context> ContextBuilder::build() const
{
  return ucxx::createContext(_impl->configMap, _impl->featureFlags);
}

}  // namespace experimental

}  // namespace ucxx
