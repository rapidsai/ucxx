/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <ucxx/buffer.h>
#include <ucxx/delayed_submission.h>
#include <ucxx/internal/request_am.h>
#include <ucxx/request_am.h>

namespace ucxx {

namespace internal {

RecvAmMessage::RecvAmMessage(internal::AmData* amData,
                             ucp_ep_h ep,
                             std::shared_ptr<RequestAm> request,
                             std::shared_ptr<Buffer> buffer)
  : _amData(amData), _ep(ep), _request(request), _buffer(buffer)
{
  _request->_delayedSubmission = std::make_shared<DelayedSubmission>(
    TransferDirection::Receive,
    _buffer->data(),
    _buffer->getSize(),
    DelayedSubmissionData(
      DelayedSubmissionOperationType::Am, TransferDirection::Receive, std::monostate{}));
}

void RecvAmMessage::setUcpRequest(void* request) { _request->_request = request; }

void RecvAmMessage::callback(void* request, ucs_status_t status)
{
  _request->_buffer = _buffer;
  _request->callback(request, status);
  {
    std::lock_guard<std::mutex> lock(_amData->_mutex);
    _amData->_recvAmMessageMap.erase(_request.get());
  }
}

}  // namespace internal

}  // namespace ucxx
