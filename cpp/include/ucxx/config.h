/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

#include <ucxx/utils.h>

namespace ucxx {

class UCPConfig {
 private:
  ucp_config_t* _handle{nullptr};
  std::map<std::string, std::string> _config_dict;

 public:
  UCPConfig() = default;

  UCPConfig(std::map<std::string, std::string>& ucx_config)
  {
    this->_handle = _read_ucx_config(ucx_config);
  }

  ~UCPConfig()
  {
    if (this->_handle != nullptr) ucp_config_release(this->_handle);
  }

  UCPConfig(const UCPConfig&) = delete;
  UCPConfig& operator=(UCPConfig const&) = delete;

  UCPConfig(UCPConfig&& o) noexcept : _handle{std::exchange(o._handle, nullptr)} {}

  UCPConfig& operator=(UCPConfig&& o) noexcept
  {
    this->_handle = std::exchange(o._handle, nullptr);

    return *this;
  }

  std::map<std::string, std::string> get()
  {
    if (this->_config_dict.empty()) this->_config_dict = ucx_config_to_dict(_handle);
    return this->_config_dict;
  }

  ucp_config_t* get_handle() { return _handle; }
};

}  // namespace ucxx
