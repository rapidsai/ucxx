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

#if UCXX_ENABLE_PYTHON
#include <ucxx/python/future.h>
#endif

namespace ucxx {

class UCXXRequestStream : public UCXXRequest {
 private:
  UCXXRequestStream(std::shared_ptr<UCXXEndpoint> endpoint, bool send, void* buffer, size_t length);

 public:
  friend std::shared_ptr<UCXXRequestStream> createRequestStream(
    std::shared_ptr<UCXXEndpoint> endpoint, bool send, void* buffer, size_t length);

  virtual void populateNotificationRequest();

  void request();

  static void stream_send_callback(void* request, ucs_status_t status, void* arg);

  static void stream_recv_callback(void* request, ucs_status_t status, size_t length, void* arg);
};

}  // namespace ucxx
