/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <ucxx/buffer.h>
#include <ucxx/delayed_submission.h>
#include <ucxx/internal/request_am.h>
#include <ucxx/request_am.h>
#include <ucxx/typedefs.h>

#include <memory>

namespace ucxx {

namespace internal {

RecvAmMessage::RecvAmMessage(internal::AmData* amData,
                             ucp_ep_h ep,
                             std::shared_ptr<RequestAm> request,
                             std::shared_ptr<Buffer> buffer,
                             AmReceiverCallbackType receiverCallback)
  : _amData(amData), _ep(ep), _request(request)
{
  std::visit(data::dispatch{
               [this, buffer](data::AmReceive& amReceive) { amReceive._buffer = buffer; },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _request->_requestData);

  if (receiverCallback) {
    _request->_callback = [this, receiverCallback](ucs_status_t, std::shared_ptr<void>) {
      receiverCallback(_request);
    };
  }
}

void RecvAmMessage::setUcpRequest(void* request) { _request->_request = request; }

void RecvAmMessage::callback(void* request, ucs_status_t status)
{
  std::visit(data::dispatch{
               [this, request, status](data::AmReceive amReceive) {
                 _request->callback(request, status);
                 {
                   std::lock_guard<std::mutex> lock(_amData->_mutex);
                   _amData->_recvAmMessageMap.erase(_request.get());
                 }
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _request->_requestData);
}

}  // namespace internal

}  // namespace ucxx
