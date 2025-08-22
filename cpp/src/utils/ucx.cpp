/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <string>
#include <utility>

#include <ucs/type/status.h>
#include <ucxx/exception.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

namespace utils {

void ucsErrorThrow(const ucs_status_t status, const std::string& userMessage)
{
  std::string message = userMessage.empty() ? ucs_status_string(status) : std::move(userMessage);

  switch (status) {
    case UCS_OK: return;
    case UCS_INPROGRESS: return;
    case UCS_ERR_NO_MESSAGE: throw ucxx::NoMessageError(message); return;
    case UCS_ERR_NO_RESOURCE: throw ucxx::NoResourceError(message); return;
    case UCS_ERR_IO_ERROR: throw ucxx::IOError(message); return;
    case UCS_ERR_NO_MEMORY: throw ucxx::NoMemoryError(message); return;
    case UCS_ERR_INVALID_PARAM: throw ucxx::InvalidParamError(message); return;
    case UCS_ERR_UNREACHABLE: throw ucxx::UnreachableError(message); return;
    case UCS_ERR_INVALID_ADDR: throw ucxx::InvalidAddrError(message); return;
    case UCS_ERR_NOT_IMPLEMENTED: throw ucxx::NotImplementedError(message); return;
    case UCS_ERR_MESSAGE_TRUNCATED: throw ucxx::MessageTruncatedError(message); return;
    case UCS_ERR_NO_PROGRESS: throw ucxx::NoProgressError(message); return;
    case UCS_ERR_BUFFER_TOO_SMALL: throw ucxx::BufferTooSmallError(message); return;
    case UCS_ERR_NO_ELEM: throw ucxx::NoElemError(message); return;
    case UCS_ERR_SOME_CONNECTS_FAILED: throw ucxx::SomeConnectsFailedError(message); return;
    case UCS_ERR_NO_DEVICE: throw ucxx::NoDeviceError(message); return;
    case UCS_ERR_BUSY: throw ucxx::BusyError(message); return;
    case UCS_ERR_CANCELED: throw ucxx::CanceledError(message); return;
    case UCS_ERR_SHMEM_SEGMENT: throw ucxx::ShmemSegmentError(message); return;
    case UCS_ERR_ALREADY_EXISTS: throw ucxx::AlreadyExistsError(message); return;
    case UCS_ERR_OUT_OF_RANGE: throw ucxx::OutOfRangeError(message); return;
    case UCS_ERR_TIMED_OUT: throw ucxx::TimedOutError(message); return;
    case UCS_ERR_EXCEEDS_LIMIT: throw ucxx::ExceedsLimitError(message); return;
    case UCS_ERR_UNSUPPORTED: throw ucxx::UnsupportedError(message); return;
    case UCS_ERR_REJECTED: throw ucxx::RejectedError(message); return;
    case UCS_ERR_NOT_CONNECTED: throw ucxx::NotConnectedError(message); return;
    case UCS_ERR_CONNECTION_RESET: throw ucxx::ConnectionResetError(message); return;
    case UCS_ERR_FIRST_LINK_FAILURE: throw ucxx::FirstLinkFailureError(message); return;
    case UCS_ERR_LAST_LINK_FAILURE: throw ucxx::LastLinkFailureError(message); return;
    case UCS_ERR_FIRST_ENDPOINT_FAILURE: throw ucxx::FirstEndpointFailureError(message); return;
    case UCS_ERR_ENDPOINT_TIMEOUT: throw ucxx::EndpointTimeoutError(message); return;
    case UCS_ERR_LAST_ENDPOINT_FAILURE: throw ucxx::LastEndpointFailureError(message); return;
    default: throw ucxx::Error(message); return;
  }
}

}  // namespace utils

}  // namespace ucxx
