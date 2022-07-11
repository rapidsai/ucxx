/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <cstdint>
#include <cstring>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/config.h>
#include <ucxx/constructors.h>

namespace ucxx {

class Worker;

class Context : public Component {
 private:
  ucp_context_h _handle{nullptr};
  Config _config{};
  uint64_t _featureFlags{0};
  bool _cudaSupport{false};

  Context(const ConfigMap ucx_config, const uint64_t featureFlags);

 public:
  static constexpr uint64_t defaultFeatureFlags =
    UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP | UCP_FEATURE_STREAM | UCP_FEATURE_AM | UCP_FEATURE_RMA;

  Context()               = delete;
  Context(const Context&) = delete;
  Context& operator=(Context const&) = delete;
  Context(Context&& o)               = delete;
  Context& operator=(Context&& o) = delete;

  friend std::shared_ptr<Context> createContext(ConfigMap ucxConfig, const uint64_t featureFlags);

  ~Context();

  ConfigMap getConfig();

  ucp_context_h getHandle();

  std::string getInfo();

  uint64_t getFeatureFlags() const;

  std::shared_ptr<Worker> createWorker(const bool enableDelayedSubmission = false,
                                       const bool enablePythonFuture      = false);
};

}  // namespace ucxx
