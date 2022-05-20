/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

#include <ucxx/notification_request.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>

namespace ucxx {

class UCXXRequestTag : public UCXXRequest {
 private:
  UCXXRequestTag(std::shared_ptr<UCXXEndpoint> endpoint,
                 bool send,
                 void* buffer,
                 size_t length,
                 ucp_tag_t tag,
                 const bool enablePythonFuture                               = true,
                 std::function<void(std::shared_ptr<void>)> callbackFunction = nullptr,
                 std::shared_ptr<void> callbackData                          = nullptr);

 public:
  friend std::shared_ptr<UCXXRequestTag> createRequestTag(
    std::shared_ptr<UCXXEndpoint> endpoint,
    bool send,
    void* buffer,
    size_t length,
    ucp_tag_t tag,
    const bool enablePythonFuture,
    std::function<void(std::shared_ptr<void>)> callbackFunction,
    std::shared_ptr<void> callbackData);

  virtual void populateNotificationRequest();

  void request();

  static void tag_send_callback(void* request, ucs_status_t status, void* arg);

  static void tag_recv_callback(void* request,
                                ucs_status_t status,
                                const ucp_tag_recv_info_t* info,
                                void* arg);
};

}  // namespace ucxx
