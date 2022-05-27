/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

namespace ucxx {

class UCXXConfig {
 private:
  ucp_config_t* _handle{nullptr};
  UCXXConfigMap _configMap;

  ucp_config_t* readUCXConfig(UCXXConfigMap userOptions);

  UCXXConfigMap ucxConfigToMap();

 public:
  UCXXConfig() = default;

  UCXXConfig(const UCXXConfig&) = delete;
  UCXXConfig& operator=(UCXXConfig const&) = delete;
  UCXXConfig(UCXXConfig&& o)               = delete;
  UCXXConfig& operator=(UCXXConfig&& o) = delete;

  UCXXConfig(UCXXConfigMap userOptions);

  ~UCXXConfig();

  UCXXConfigMap get();

  ucp_config_t* get_handle();
};

}  // namespace ucxx
