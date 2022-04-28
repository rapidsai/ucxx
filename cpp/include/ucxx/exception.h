/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <exception>
#include <string>

namespace ucxx {

class UCXXError : public std::exception {
 private:
  std::string _msg{};

 public:
  UCXXError(const std::string& msg) : _msg{msg} {}

  virtual const char* what() const noexcept override { return this->_msg.c_str(); }
};

class UCXXCanceledError : public UCXXError {
 public:
  UCXXCanceledError(const std::string& msg) : UCXXError(msg) {}
};

class UCXXConfigError : public UCXXError {
 public:
  UCXXConfigError(const std::string& msg) : UCXXError(msg) {}
};

class UCXXConnectionResetError : public UCXXError {
 public:
  UCXXConnectionResetError(const std::string& msg) : UCXXError(msg) {}
};

}  // namespace ucxx
