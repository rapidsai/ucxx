/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/buffer.h>
#include <ucxx/endpoint.h>
#include <ucxx/future.h>
#include <ucxx/request.h>

namespace ucxx {

class RequestTagMulti;

struct BufferRequest {
  std::shared_ptr<Request> request{nullptr};  ///< The `ucxx::RequestTag` of a header or frame
  std::shared_ptr<std::string> stringBuffer{nullptr};  ///< Serialized `Header`
  std::shared_ptr<Buffer> buffer{nullptr};  ///< Internally allocated buffer to receive a frame

  BufferRequest();
  ~BufferRequest();

  BufferRequest(const BufferRequest&)            = delete;
  BufferRequest& operator=(BufferRequest const&) = delete;
  BufferRequest(BufferRequest&& o)               = delete;
  BufferRequest& operator=(BufferRequest&& o)    = delete;
};

typedef std::shared_ptr<BufferRequest> BufferRequestPtr;

class RequestTagMulti : public Request {
 private:
  bool _send{false};       ///< Whether this is a send (`true`) operation or recv (`false`)
  ucp_tag_t _tag{0};       ///< Tag to match
  size_t _totalFrames{0};  ///< The total number of frames handled by this request
  std::mutex
    _completedRequestsMutex{};   ///< Mutex to control access to completed requests container
  size_t _completedRequests{0};  ///< Count requests that already completed

 public:
  std::vector<BufferRequestPtr> _bufferRequests{};  ///< Container of all requests posted
  bool _isFilled{false};                            ///< Whether the all requests have been posted

 private:
  RequestTagMulti()                                  = delete;
  RequestTagMulti(const RequestTagMulti&)            = delete;
  RequestTagMulti& operator=(RequestTagMulti const&) = delete;
  RequestTagMulti(RequestTagMulti&& o)               = delete;
  RequestTagMulti& operator=(RequestTagMulti&& o)    = delete;

  /**
   * @brief Protected constructor of a multi-buffer tag receive request.
   *
   * Construct multi-buffer tag receive request, registering the request to the
   * `std::shared_ptr<Endpoint>` parent so that it may be canceled if necessary. This
   * constructor is responsible for creating a Python future that can be later awaited
   * in Python asynchronous code, which is indenpendent of the Python futures used by
   * the underlying `ucxx::RequestTag` object, which will be invisible to the user. Once
   * the initial setup is complete, `callback()` is called to initiate receiving by posting
   * the first request to receive a header.
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component
   * @param[in] send                whether this is a send (`true`) or receive (`false`)
   *                                tag request.
   * @param[in] tag                 the tag to match.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   */
  RequestTagMulti(std::shared_ptr<Endpoint> endpoint,
                  const bool send,
                  const ucp_tag_t tag,
                  const bool enablePythonFuture);

  /**
   * @brief Receive all frames.
   *
   * Once the header(s) has(have) been received, receiving frames containing the actual data
   * is the next step. This method parses the header(s) and creates as many
   * `ucxx::RequestTag` objects as necessary, each one that will handle a single sending or
   * receiving a single frame.
   *
   * Finally, the object is marked as filled, meaning that all requests were already
   * scheduled and are waiting for completion.
   *
   * @throws std::runtime_error if called by a send request.
   */
  void recvFrames();

  /**
   * @brief Receive a message with header.
   *
   * Create the request to receive a message with header, setting
   * `ucxx::RequestTagMulti::callback` as the user-defined callback of `ucxx::RequestTag` to
   * handle the next request.
   *
   * @throws std::runtime_error if called by a send request.
   */
  void recvHeader();

  /**
   * @brief Send all header(s) and frame(s).
   *
   * Build header request(s) and send them, followed by requests to send all frame(s).
   *
   * @throws std::length_error  if the lengths of `buffer`, `size` and `isCUDA` do not
   *                            match.
   */
  void send(const std::vector<void*>& buffer,
            const std::vector<size_t>& size,
            const std::vector<int>& isCUDA);

 public:
  /**
   * @brief Enqueue a multi-buffer tag send operation.
   *
   * Initiate a multi-buffer tag send operation, returning a
   * `std::shared<ucxx::RequestTagMulti>` that can be later awaited and checked for errors.
   * This is a non-blocking operation, and the status of the transfer must be verified from
   * the resulting request object before the data can be released.
   *
   * The primary use of multi-buffer transfers is in Python where we want to reduce the
   * amount of futures needed to watch for, thus reducing Python overhead. However, this
   * may be used as a convenience implementation for transfers that require multiple
   * frames, internally this is implemented as one or more `ucxx::RequestTag` calls sending
   * headers (depending on the number of frames being transferred), followed by one
   * `ucxx::RequestTag` for each data frame.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX to be compiled with
   * `UCXX_ENABLE_PYTHON=1`.
   *
   * @throws  std::runtime_error  if sizes of `buffer`, `size` and `isCUDA` do not match.
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component
   * @param[in] tag                 the tag to match.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  friend std::shared_ptr<RequestTagMulti> createRequestTagMultiSend(
    std::shared_ptr<Endpoint> endpoint,
    const std::vector<void*>& buffer,
    const std::vector<size_t>& size,
    const std::vector<int>& isCUDA,
    const ucp_tag_t tag,
    const bool enablePythonFuture);

  /**
   * @brief Enqueue a multi-buffer tag receive operation.
   *
   * Enqueue a multi-buffer tag receive operation, returning a
   * `std::shared<ucxx::RequestTagMulti>` that can be later awaited and checked for errors.
   * This is a non-blocking operation, and because the receiver has no a priori knowledge
   * of the data being received, memory allocations are automatically handled internally.
   * The receiver must have the same capabilities of the sender, so that if the sender is
   * compiled with RMM support to allow for CUDA transfers, the receiver must have the
   * ability to understand and allocate CUDA memory.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX to be compiled with
   * `UCXX_ENABLE_PYTHON=1`.
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component
   * @param[in] tag                 the tag to match.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  friend std::shared_ptr<RequestTagMulti> createRequestTagMultiRecv(
    std::shared_ptr<Endpoint> endpoint, const ucp_tag_t tag, const bool enablePythonFuture);

  /**
   * @brief `ucxx::RequestTagMulti` destructor.
   *
   * Free internal resources.
   */
  virtual ~RequestTagMulti();

  /**
   * @brief Mark request as completed.
   *
   * Mark a single `ucxx::RequestTag` as completed. This method is passed as the
   * user-defined callback to the `ucxx::RequestTag` constructor, which will then be
   * executed when that completes.
   *
   * When this method is called, the request that completed will be pushed into a container
   * which will be later used to evaluate if all frames completed and set the final status
   * of the multi-transfer request and the Python future, if enabled. The final status is
   * either `UCS_OK` if all underlying requests completed successfully, otherwise it will
   * contain the status of the first failing request, for granular information the user
   * may still verify each of the underlying requests individually.
   *
   * @param[in] status the status of the request being completed.
   * @param[in] request the `ucxx::BufferRequest` object containing a single tag .
   */
  void markCompleted(ucs_status_t status, RequestCallbackUserData request);

  /**
   * @brief Callback to submit request to receive new header or frames.
   *
   * When a receive multi-transfer tag request is created or has received a new header, this
   * callback must be executed to ensure the next request to receive is submitted.
   *
   * If no requests for the present `ucxx::RequestTagMulti` transfer have been posted yet,
   * create one receiving a message with header. If the previous received request is header
   * containing the `next` flag set, then the next request is another header. Otherwise, the
   * next incoming message(s) is(are) frame(s).
   *
   * When called, the callback receives a single argument, the status of the current request.
   *
   * @param[in] status the status of the request being completed.
   * @throws std::runtime_error if called by a send request.
   */
  void recvCallback(ucs_status_t status);

  void populateDelayedSubmission() override;

  void cancel() override;
};

typedef std::shared_ptr<RequestTagMulti> RequestTagMultiPtr;

}  // namespace ucxx
