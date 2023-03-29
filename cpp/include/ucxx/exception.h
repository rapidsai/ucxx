/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <exception>
#include <string>

namespace ucxx {

class Error : public std::exception {
 private:
  std::string _msg{};

 public:
  explicit Error(const std::string& msg) : _msg{msg} {}

  const char* what() const noexcept override { return this->_msg.c_str(); }
};

/**
 * UCS_ERR_NO_MESSAGE
 */
class NoMessageError : public Error {
 public:
  explicit NoMessageError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NO_RESOURCE
 */
class NoResourceError : public Error {
 public:
  explicit NoResourceError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_IO_ERROR
 */
class IOError : public Error {
 public:
  explicit IOError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NO_MEMORY
 */
class NoMemoryError : public Error {
 public:
  explicit NoMemoryError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_INVALID_PARAM
 */
class InvalidParamError : public Error {
 public:
  explicit InvalidParamError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_UNREACHABLE
 */
class UnreachableError : public Error {
 public:
  explicit UnreachableError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_INVALID_ADDR
 */
class InvalidAddrError : public Error {
 public:
  explicit InvalidAddrError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NOT_IMPLEMENTED
 */
class NotImplementedError : public Error {
 public:
  explicit NotImplementedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_MESSAGE_TRUNCATED
 */
class MessageTruncatedError : public Error {
 public:
  explicit MessageTruncatedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NO_PROGRESS
 */
class NoProgressError : public Error {
 public:
  explicit NoProgressError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_BUFFER_TOO_SMALL
 */
class BufferTooSmallError : public Error {
 public:
  explicit BufferTooSmallError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NO_ELEM
 */
class NoElemError : public Error {
 public:
  explicit NoElemError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_SOME_CONNECTS_FAILED
 */
class SomeConnectsFailedError : public Error {
 public:
  explicit SomeConnectsFailedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NO_DEVICE
 */
class NoDeviceError : public Error {
 public:
  explicit NoDeviceError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_BUSY
 */
class BusyError : public Error {
 public:
  explicit BusyError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_CANCELED
 */
class CanceledError : public Error {
 public:
  explicit CanceledError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_SHMEM_SEGMENT
 */
class ShmemSegmentError : public Error {
 public:
  explicit ShmemSegmentError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_ALREADY_EXISTS
 */
class AlreadyExistsError : public Error {
 public:
  explicit AlreadyExistsError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_OUT_OF_RANGE
 */
class OutOfRangeError : public Error {
 public:
  explicit OutOfRangeError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_TIMED_OUT
 */
class TimedOutError : public Error {
 public:
  explicit TimedOutError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_EXCEEDS_LIMIT
 */
class ExceedsLimitError : public Error {
 public:
  explicit ExceedsLimitError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_UNSUPPORTED
 */
class UnsupportedError : public Error {
 public:
  explicit UnsupportedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_REJECTED
 */
class RejectedError : public Error {
 public:
  explicit RejectedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_NOT_CONNECTED
 */
class NotConnectedError : public Error {
 public:
  explicit NotConnectedError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_CONNECTION_RESET
 */
class ConnectionResetError : public Error {
 public:
  explicit ConnectionResetError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_FIRST_LINK_FAILURE
 */
class FirstLinkFailureError : public Error {
 public:
  explicit FirstLinkFailureError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_LAST_LINK_FAILURE
 */
class LastLinkFailureError : public Error {
 public:
  explicit LastLinkFailureError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_FIRST_ENDPOINT_FAILURE
 */
class FirstEndpointFailureError : public Error {
 public:
  explicit FirstEndpointFailureError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_ENDPOINT_TIMEOUT
 */
class EndpointTimeoutError : public Error {
 public:
  explicit EndpointTimeoutError(const std::string& msg) : Error(msg) {}
};

/**
 * UCS_ERR_LAST_ENDPOINT_FAILURE
 */
class LastEndpointFailureError : public Error {
 public:
  explicit LastEndpointFailureError(const std::string& msg) : Error(msg) {}
};

}  // namespace ucxx
