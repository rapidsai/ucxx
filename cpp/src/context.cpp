/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <cstring>
#include <iostream>

#include <ucxx/context.h>
#include <ucxx/log.h>
#include <ucxx/utils/file_descriptor.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

Context::Context(const ConfigMap ucxConfig, const uint64_t featureFlags)
  : _config{ucxConfig}, _featureFlags{featureFlags}
{
  ucp_params_t params;

  parseLogLevel();

  // UCP
  std::memset(&params, 0, sizeof(params));
  params.field_mask = UCP_PARAM_FIELD_FEATURES;
  params.features   = featureFlags;

  utils::ucsErrorThrow(ucp_init(&params, this->_config.getHandle(), &this->_handle));

  // UCX supports CUDA if "cuda" is part of the TLS or TLS is "all"
  auto configMap = this->_config.get();
  auto tls       = configMap.find("TLS");
  if (tls != configMap.end()) {
    if (!tls->second.empty() and tls->second[0] == '^') {
      this->_cudaSupport = tls->second.find("cuda") == std::string::npos;
    } else {
      this->_cudaSupport = tls->second == "all" || tls->second.find("cuda") != std::string::npos;
    }
  }

  ucxx_info("UCP initiated using config: ");
  for (const auto& kv : configMap)
    ucxx_info("  %s: %s", kv.first.c_str(), kv.second.c_str());
}

std::shared_ptr<Context> createContext(const ConfigMap ucxConfig, const uint64_t featureFlags)
{
  return std::shared_ptr<Context>(new Context(ucxConfig, featureFlags));
}

Context::~Context()
{
  if (this->_handle != nullptr) ucp_cleanup(this->_handle);
}

ConfigMap Context::getConfig() { return this->_config.get(); }

ucp_context_h Context::getHandle() { return this->_handle; }

std::string Context::getInfo()
{
  FILE* TextFileDescriptor = utils::createTextFileDescriptor();
  ucp_context_print_info(this->_handle, TextFileDescriptor);
  return utils::decodeTextFileDescriptor(TextFileDescriptor);
}

uint64_t Context::getFeatureFlags() const { return _featureFlags; }

std::shared_ptr<Worker> Context::createWorker(const bool enableDelayedSubmission,
                                              const bool enablePythonFuture)
{
  auto context = std::dynamic_pointer_cast<Context>(shared_from_this());
  auto worker  = ucxx::createWorker(context, enableDelayedSubmission, enablePythonFuture);
  return worker;
}

}  // namespace ucxx
