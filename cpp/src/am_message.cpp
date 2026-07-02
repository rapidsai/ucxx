/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>

#include <ucp/api/ucp.h>

#include <ucxx/am_message.h>
#include <ucxx/worker.h>

namespace ucxx {

AmMessage::AmMessage(Worker* worker,
                     ucp_ep_h replyEp,
                     const void* header,
                     size_t headerLength,
                     void* data,
                     size_t length,
                     const ucp_am_recv_param_t* param)
  : _worker(worker),
    _replyEp(replyEp),
    _header(header),
    _headerLength(headerLength),
    _data(data),
    _length(length),
    _param(param)
{
}

const void* AmMessage::header() const noexcept { return _header; }

size_t AmMessage::headerLength() const noexcept { return _headerLength; }

size_t AmMessage::length() const noexcept { return _length; }

bool AmMessage::isRendezvous() const noexcept
{
  return (_param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) != 0;
}

ucp_ep_h AmMessage::replyEp() const noexcept { return _replyEp; }

const void* AmMessage::data() const noexcept { return isRendezvous() ? nullptr : _data; }

std::shared_ptr<Request> AmMessage::receive(void* buffer,
                                            size_t count,
                                            ucp_datatype_t datatype,
                                            bool enablePythonFuture,
                                            RequestCallbackUserFunction callbackFunction,
                                            RequestCallbackUserData callbackData)
{
  _receiveCalled = true;
  return _worker->amRecvData(
    _data, buffer, count, datatype, enablePythonFuture, callbackFunction, callbackData);
}

void AmMessage::reject(ucs_status_t reason)
{
  _rejected     = true;
  _rejectStatus = reason;
}

}  // namespace ucxx
