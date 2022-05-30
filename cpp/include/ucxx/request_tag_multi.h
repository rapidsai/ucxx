/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/endpoint.h>
#include <ucxx/request.h>

#include <ucxx/buffer_helper.h>
#include <ucxx/request_helper.h>

#if UCXX_ENABLE_PYTHON
#include <ucxx/python/python_future.h>
#endif

namespace ucxx {

struct BufferRequest {
  std::shared_ptr<Request> request{nullptr};
  std::shared_ptr<std::string> stringBuffer{nullptr};
  std::unique_ptr<PyBuffer> pyBuffer{nullptr};
};

typedef std::shared_ptr<BufferRequest> BufferRequestPtr;

class RequestTagMulti : public std::enable_shared_from_this<RequestTagMulti> {
 private:
  std::shared_ptr<Endpoint> _endpoint = nullptr;
  bool _send{false};
  ucp_tag_t _tag{0};
  size_t _totalFrames{0};
  std::mutex _completedRequestsMutex;
  std::vector<BufferRequest*> _completedRequests{};
  ucs_status_t _status{UCS_INPROGRESS};
#if UCXX_ENABLE_PYTHON
  std::shared_ptr<PythonFuture> _pythonFuture;
#endif

 public:
  std::vector<BufferRequestPtr> _bufferRequests{};
  bool _isFilled{false};

 private:
  RequestTagMulti() = delete;

  // Recv Constructor
  RequestTagMulti(std::shared_ptr<Endpoint> endpoint, const ucp_tag_t tag);

  // Send Constructor
  RequestTagMulti(std::shared_ptr<Endpoint> endpoint,
                  std::vector<void*>& buffer,
                  std::vector<size_t>& size,
                  std::vector<int>& isCUDA,
                  const ucp_tag_t tag);

 public:
  friend std::shared_ptr<RequestTagMulti> tagMultiSend(std::shared_ptr<Endpoint> endpoint,
                                                       std::vector<void*>& buffer,
                                                       std::vector<size_t>& size,
                                                       std::vector<int>& isCUDA,
                                                       const ucp_tag_t tag);

  friend std::shared_ptr<RequestTagMulti> tagMultiRecv(std::shared_ptr<Endpoint> endpoint,
                                                       const ucp_tag_t tag);

  friend std::vector<std::unique_ptr<PyBuffer>> tagMultiRecvBlocking(
    std::shared_ptr<Endpoint> endpoint, ucp_tag_t tag);

  friend void tagMultiSendBlocking(std::shared_ptr<Endpoint> endpoint,
                                   std::vector<void*>& buffer,
                                   std::vector<size_t>& size,
                                   std::vector<int>& isCUDA,
                                   ucp_tag_t tag);

  void recvFrames();

  void markCompleted(std::shared_ptr<void> request);

  void recvHeader();

  void callback(std::shared_ptr<void> arg = nullptr);

  void send(std::vector<void*>& buffer, std::vector<size_t>& size, std::vector<int>& isCUDA);

  ucs_status_t getStatus();

  PyObject* getPyFuture();

  void checkError();

  template <typename Rep, typename Period>
  bool isCompleted(std::chrono::duration<Rep, Period> period);

  bool isCompleted(int64_t periodNs = 0);
};

typedef std::shared_ptr<RequestTagMulti> RequestTagMultiPtr;

}  // namespace ucxx
