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

class ConfigError : public Error {
 public:
  ConfigError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NO_MESSAGE
 */
class NoMessageError : public Error {
 public:
  NoMessageError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NO_RESOURCE
 */
class NoResourceError : public Error {
 public:
  NoResourceError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_IO_ERROR
 */
class IOError : public Error {
 public:
  IOError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NO_MEMORY
 */
class NoMemoryError : public Error {
 public:
  NoMemoryError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_INVALID_PARAM
 */
class InvalidParamError : public Error {
 public:
  InvalidParamError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_UNREACHABLE
 */
class UnreachableError : public Error {
 public:
  UnreachableError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_INVALID_ADDR
 */
class InvalidAddrError : public Error {
 public:
  InvalidAddrError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NOT_IMPLEMENTED
 */
class NotImplementedError : public Error {
 public:
  NotImplementedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_MESSAGE_TRUNCATED
 */
class MessageTruncatedError : public Error {
 public:
  MessageTruncatedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NO_PROGRESS
 */
class NoProgressError : public Error {
 public:
  NoProgressError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_BUFFER_TOO_SMALL
 */
class BufferTooSmallError : public Error {
 public:
  BufferTooSmallError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NO_ELEM
 */
class NoElemError : public Error {
 public:
  NoElemError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_SOME_CONNECTS_FAILED
 */
class SomeConnectsFailedError : public Error {
 public:
  SomeConnectsFailedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NO_DEVICE
 */
class NoDeviceError : public Error {
 public:
  NoDeviceError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_BUSY
 */
class BusyError : public Error {
 public:
  BusyError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_CANCELED
 */
class CanceledError : public Error {
 public:
  CanceledError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_SHMEM_SEGMENT
 */
class ShmemSegmentError : public Error {
 public:
  ShmemSegmentError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_ALREADY_EXISTS
 */
class AlreadyExistsError : public Error {
 public:
  AlreadyExistsError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_OUT_OF_RANGE
 */
class OutOfRangeError : public Error {
 public:
  OutOfRangeError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_TIMED_OUT
 */
class TimedOutError : public Error {
 public:
  TimedOutError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_EXCEEDS_LIMIT
 */
class ExceedsLimitError : public Error {
 public:
  ExceedsLimitError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_UNSUPPORTED
 */
class UnsupportedError : public Error {
 public:
  UnsupportedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_REJECTED
 */
class RejectedError : public Error {
 public:
  RejectedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NOT_CONNECTED
 */
class NotConnectedError : public Error {
 public:
  NotConnectedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_CONNECTION_RESET
 */
class ConnectionResetError : public Error {
 public:
  ConnectionResetError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_FIRST_LINK_FAILURE
 */
class FirstLinkFailureError : public Error {
 public:
  FirstLinkFailureError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_LAST_LINK_FAILURE
 */
class LastLinkFailureError : public Error {
 public:
  LastLinkFailureError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_FIRST_ENDPOINT_FAILURE
 */
class FirstEndpointFailureError : public Error {
 public:
  FirstEndpointFailureError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_ENDPOINT_TIMEOUT
 */
class EndpointTimeoutError : public Error {
 public:
  EndpointTimeoutError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_LAST_ENDPOINT_FAILURE
 */
class LastEndpointFailureError : public Error {
 public:
  LastEndpointFailureError(const std::string& msg) : Error(msg) {}
};

}  // namespace ucxx
