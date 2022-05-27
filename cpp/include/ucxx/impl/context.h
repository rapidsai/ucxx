/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <cstring>
#include <iostream>

#include <ucxx/context.h>
#include <ucxx/initializer.h>
#include <ucxx/utils.h>

namespace ucxx {

UCXXContext::UCXXContext(const UCXXConfigMap ucx_config, const uint64_t feature_flags)
  : _config{UCXXConfig(ucx_config)}, _feature_flags{feature_flags}
{
  ucp_params_t ucp_params;

  ucxx::UCXXInitializer::getInstance();

  // UCP
  std::memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
  ucp_params.features   = feature_flags;

  assert_ucs_status(ucp_init(&ucp_params, this->_config.get_handle(), &this->_handle));

  // UCX supports CUDA if "cuda" is part of the TLS or TLS is "all"
  auto config_map = this->_config.get();
  auto tls        = config_map.find("TLS");
  if (tls != config_map.end())
    this->_cuda_support = tls->second == "all" || tls->second.find("cuda") != std::string::npos;

  ucxx_info("UCP initiated using config: ");
  for (const auto& kv : config_map)
    ucxx_info("  %s: %s", kv.first.c_str(), kv.second.c_str());
}

std::shared_ptr<UCXXContext> createContext(const UCXXConfigMap ucx_config,
                                           const uint64_t feature_flags)
{
  return std::shared_ptr<UCXXContext>(new UCXXContext(ucx_config, feature_flags));
}

UCXXContext::~UCXXContext()
{
  if (this->_handle != nullptr) ucp_cleanup(this->_handle);
}

UCXXConfigMap UCXXContext::get_config() { return this->_config.get(); }

ucp_context_h UCXXContext::get_handle() { return this->_handle; }

std::string UCXXContext::get_info()
{
  FILE* text_fd = create_text_fd();
  ucp_context_print_info(this->_handle, text_fd);
  return decode_text_fd(text_fd);
}

uint64_t UCXXContext::get_feature_flags() const { return _feature_flags; }

std::shared_ptr<UCXXWorker> UCXXContext::createWorker(const bool enableDelayedNotification)
{
  auto context = std::dynamic_pointer_cast<UCXXContext>(shared_from_this());
  auto worker  = ucxx::createWorker(context, enableDelayedNotification);
  return worker;
}

}  // namespace ucxx
