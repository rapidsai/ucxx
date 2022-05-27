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

class UCXXWorker;

class UCXXContext : public UCXXComponent {
 private:
  ucp_context_h _handle{nullptr};
  UCXXConfig _config{};
  uint64_t _feature_flags{0};
  bool _cuda_support{false};

  UCXXContext(const UCXXConfigMap ucx_config, const uint64_t feature_flags);

 public:
  static constexpr uint64_t default_feature_flags =
    UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP | UCP_FEATURE_STREAM | UCP_FEATURE_AM | UCP_FEATURE_RMA;

  UCXXContext()                   = delete;
  UCXXContext(const UCXXContext&) = delete;
  UCXXContext& operator=(UCXXContext const&) = delete;
  UCXXContext(UCXXContext&& o)               = delete;
  UCXXContext& operator=(UCXXContext&& o) = delete;

  friend std::shared_ptr<UCXXContext> createContext(UCXXConfigMap ucx_config,
                                                    const uint64_t feature_flags);

  ~UCXXContext();

  UCXXConfigMap get_config();

  ucp_context_h get_handle();

  std::string get_info();

  uint64_t get_feature_flags() const;

  std::shared_ptr<UCXXWorker> createWorker(const bool enableDelayedNotification = true);
};

}  // namespace ucxx
