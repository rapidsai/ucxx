/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

#include <ucxx/utils.h>

namespace ucxx {

typedef std::unordered_map<std::string, std::string> UCXXConfigMap;

class UCXXConfig {
 private:
  ucp_config_t* _handle{nullptr};
  UCXXConfigMap _configMap;

  ucp_config_t* readUCXConfig(UCXXConfigMap userOptions)
  {
    ucs_status_t status;
    std::string status_msg;

    status = ucp_config_read(NULL, NULL, &_handle);
    if (status != UCS_OK) {
      status_msg = ucs_status_string(status);
      throw ucxx::UCXXConfigError(std::string("Couldn't read the UCX options: ") + status_msg);
    }

    // Modify the UCX configuration options based on `config_dict`
    for (const auto& kv : userOptions) {
      status = ucp_config_modify(_handle, kv.first.c_str(), kv.second.c_str());
      if (status != UCS_OK) {
        ucp_config_release(_handle);

        if (status == UCS_ERR_NO_ELEM) {
          throw ucxx::UCXXConfigError(std::string("Option ") + kv.first +
                                      std::string("doesn't exist"));
        } else {
          throw ucxx::UCXXConfigError(ucs_status_string(status));
        }
      }
    }

    return _handle;
  }

  UCXXConfigMap ucxConfigToMap()
  {
    if (_configMap.empty()) {
      FILE* text_fd = create_text_fd();
      ucp_config_print(_handle, text_fd, NULL, UCS_CONFIG_PRINT_CONFIG);
      std::istringstream text{decode_text_fd(text_fd)};

      std::string delim = "=";
      std::string line;
      while (std::getline(text, line)) {
        size_t split  = line.find(delim);
        std::string k = line.substr(4, split - 4);  // 4 to strip "UCX_" prefix
        std::string v = line.substr(split + delim.length(), std::string::npos);
        _configMap[k] = v;
      }
    }

    return _configMap;
  }

 public:
  UCXXConfig() = default;

  UCXXConfig(const UCXXConfig&) = delete;
  UCXXConfig& operator=(UCXXConfig const&) = delete;
  UCXXConfig(UCXXConfig&& o)               = delete;
  UCXXConfig& operator=(UCXXConfig&& o) = delete;

  UCXXConfig(UCXXConfigMap userOptions) { readUCXConfig(userOptions); }

  ~UCXXConfig()
  {
    if (this->_handle != nullptr) ucp_config_release(this->_handle);
  }

  UCXXConfigMap get() { return ucxConfigToMap(); }

  ucp_config_t* get_handle() { return _handle; }
};

}  // namespace ucxx
