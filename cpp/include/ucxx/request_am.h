/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <utility>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>

namespace ucxx {

class Buffer;

namespace internal {
class RecvAmMessage;
}  // namespace internal

class RequestAM : public Request {
 private:
  friend class internal::RecvAmMessage;

  ucs_memory_type_t _sendHeader{};           ///< The header to send
  std::shared_ptr<Buffer> _buffer{nullptr};  ///< The AM received message buffer

  RequestAM(std::shared_ptr<Endpoint> endpoint,
            void* buffer,
            size_t length,
            const bool enablePythonFuture                = false,
            RequestCallbackUserFunction callbackFunction = nullptr,
            RequestCallbackUserData callbackData         = nullptr);

  RequestAM(std::shared_ptr<Component> endpointOrWorker,
            const bool enablePythonFuture                = false,
            RequestCallbackUserFunction callbackFunction = nullptr,
            RequestCallbackUserData callbackData         = nullptr);

 public:
  friend std::shared_ptr<RequestAM> createRequestAMSend(
    std::shared_ptr<Endpoint> endpoint,
    void* buffer,
    size_t length,
    const bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);

  friend std::shared_ptr<RequestAM> createRequestAMRecv(
    std::shared_ptr<Component> endpointOrWorker,
    const bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);

  virtual void populateDelayedSubmission();

  void request();

  static void amSendCallback(void* request, ucs_status_t status, void* user_data);

  static ucs_status_t recvCallback(void* arg,
                                   const void* header,
                                   size_t header_length,
                                   void* data,
                                   size_t length,
                                   const ucp_am_recv_param_t* param);

  std::shared_ptr<Buffer> getRecvBuffer() override;
};

}  // namespace ucxx
