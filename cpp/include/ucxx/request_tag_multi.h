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

#include <ucxx/buffer.h>
#include <ucxx/endpoint.h>
#include <ucxx/future.h>
#include <ucxx/request.h>
#include <ucxx/request_helper.h>

namespace ucxx {

struct BufferRequest {
  std::shared_ptr<Request> request{nullptr};  ///< The `ucxx::RequestTag` of a header or frame
  std::shared_ptr<std::string> stringBuffer{nullptr};  ///< Serialized `Header`
  Buffer* buffer{nullptr};  ///< Internally allocated buffer to receive a frame
};

typedef std::shared_ptr<BufferRequest> BufferRequestPtr;

class RequestTagMulti : public std::enable_shared_from_this<RequestTagMulti> {
 private:
  std::shared_ptr<Endpoint> _endpoint{nullptr};  ///< Endpoint that generated request
  bool _send{false};       ///< Whether this is a send (`true`) operation or recv (`false`)
  ucp_tag_t _tag{0};       ///< Tag to match
  size_t _totalFrames{0};  ///< The total number of frames handled by this request
  std::mutex _completedRequestsMutex;  ///< Mutex to control access to completed requests container
  std::vector<BufferRequest*> _completedRequests{};  ///< Requests that already completed
  ucs_status_t _status{UCS_INPROGRESS};              ///< Status of the multi-buffer request
  std::shared_ptr<Future> _future;  ///< Future to be notified when transfer of all frames complete

 public:
  std::vector<BufferRequestPtr> _bufferRequests{};  ///< Container of all requests posted
  bool _isFilled{false};                            ///< Whether the all requests have been posted

 private:
  RequestTagMulti() = delete;

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
   * @param[in] tag                 the tag to match.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   */
  RequestTagMulti(std::shared_ptr<Endpoint> endpoint,
                  const ucp_tag_t tag,
                  const bool enablePythonFuture);

  /**
   * @brief Protected constructor of a multi-buffer tag send request.
   *
   * Construct multi-buffer tag send request, registering the request to the
   * `std::shared_ptr<Endpoint>` parent so that it may be canceled if necessary. This
   * constructor is responsible for creating a Python future that can be later awaited
   * in Python asynchronous code, which is indenpendent of the Python futures used by
   * the underlying `ucxx::RequestTag` object, which will be invisible to the user. Once
   * the initial setup is complete, `send()` is called to post messages containing the
   * header(s) and frame(s).
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component
   * @param[in] buffer              a vector of raw pointers to the data frames to be sent.
   * @param[in] length              a vector of size in bytes of each frame to be sent.
   * @param[in] isCUDA              a vector of booleans (integers to prevent incoherence
   *                                with other vector types) indicating whether frame is
   *                                CUDA, to ensure proper memory allocation by the
   *                                receiver.
   * @param[in] tag                 the tag to match.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   */
  RequestTagMulti(std::shared_ptr<Endpoint> endpoint,
                  std::vector<void*>& buffer,
                  std::vector<size_t>& size,
                  std::vector<int>& isCUDA,
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
  void send(std::vector<void*>& buffer, std::vector<size_t>& size, std::vector<int>& isCUDA);

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
    std::vector<void*>& buffer,
    std::vector<size_t>& size,
    std::vector<int>& isCUDA,
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
   * @brief `ucxx::RequestTagMultic` destructor.
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
   * of the multi-transfer request and the Python future, if enabled.
   *
   * @param[in] request the `ucxx::BufferRequest` object containing a single tag .
   */
  void markCompleted(std::shared_ptr<void> request);

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
   * @throws std::runtime_error if called by a send request.
   */
  void callback();

  /**
   * @brief Return the status of the request.
   *
   * Return a `ucs_status_t` that may be used for more fine-grained error handling than
   * relying on `checkError()` alone, which does not currently implement all error
   * statuses supported by UCX.
   *
   * @return the current status of the request.
   */
  ucs_status_t getStatus();

  /**
   * @brief Return the future used to check on state.
   *
   * If the object is built with Python future support, return the future that can be
   * awaited from Python, returns `nullptr` otherwise.
   *
   * @returns the Python future object or `nullptr`.
   */
  void* getFuture();

  /**
   * @brief Check whether the request completed with an error.
   *
   * Check whether the request has completed with an error, if an error occurred an
   * exception is raised, but if the request has completed or is in progress this call will
   * act as a no-op. To verify whether the request is in progress either `isCompleted()` or
   * `getStatus()` should be checked.
   *
   * @throw `ucxx::CanceledError`         if the request was canceled.
   * @throw `ucxx::MessageTruncatedError` if the message was truncated.
   * @throw `ucxx::Error`                 if another error occurred.
   */
  void checkError();

  /**
   * @brief Check whether the request has already completed.
   *
   * Check whether the request has already completed. The status of the request must be
   * verified with `getStatus()` before consumption.
   *
   * @return whether the request has completed.
   */
  bool isCompleted();
};

typedef std::shared_ptr<RequestTagMulti> RequestTagMultiPtr;

}  // namespace ucxx
