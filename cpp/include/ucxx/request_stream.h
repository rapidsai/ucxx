/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>

namespace ucxx {

class RequestStream : public Request {
 private:
  size_t _length{0};

  RequestStream(std::shared_ptr<Endpoint> endpoint,
                bool send,
                void* buffer,
                size_t length,
                const bool enablePythonFuture = false);

 public:
  friend std::shared_ptr<RequestStream> createRequestStream(std::shared_ptr<Endpoint> endpoint,
                                                            bool send,
                                                            void* buffer,
                                                            size_t length,
                                                            const bool enablePythonFuture);

  virtual void populateDelayedSubmission();

  void request();

  void callback(void* request, ucs_status_t status, size_t length);

  static void streamSendCallback(void* request, ucs_status_t status, void* arg);

  static void streamRecvCallback(void* request, ucs_status_t status, size_t length, void* arg);
};

}  // namespace ucxx
