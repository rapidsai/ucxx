/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <exception>
#include <string>

namespace ucxx {

class Error : public std::exception {
 private:
  std::string _msg{};

 public:
  Error(const std::string& msg) : _msg{msg} {}

  virtual const char* what() const noexcept override { return this->_msg.c_str(); }
};

class CanceledError : public Error {
 public:
  CanceledError(const std::string& msg) : Error(msg) {}
};

class ConfigError : public Error {
 public:
  ConfigError(const std::string& msg) : Error(msg) {}
};

class ConnectionResetError : public Error {
 public:
  ConnectionResetError(const std::string& msg) : Error(msg) {}
};

}  // namespace ucxx
