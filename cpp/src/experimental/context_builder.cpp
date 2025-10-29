/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <ucxx/context.h>
#include <ucxx/experimental/context_builder.h>

namespace ucxx {

namespace experimental {

ContextBuilder::ContextBuilder(uint64_t featureFlags) : _featureFlags(featureFlags) {}

ContextBuilder& ContextBuilder::configMap(ConfigMap configMap)
{
  _configMap = std::move(configMap);
  return *this;
}

std::shared_ptr<Context> ContextBuilder::build() const
{
  return std::shared_ptr<Context>(new Context(_configMap, _featureFlags));
}

ContextBuilder::operator std::shared_ptr<Context>() const
{
  return std::shared_ptr<Context>(new Context(_configMap, _featureFlags));
}

}  // namespace experimental

}  // namespace ucxx
