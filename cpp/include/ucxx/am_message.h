/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>

#include <ucp/api/ucp.h>

#include <ucxx/typedefs.h>

namespace ucxx {

class Request;
class Worker;

/**
 * @brief C++ view of an Active Message received by a handler registered via
 *        Worker::setAmHandler().
 *
 * An AmMessage is constructed by the AM receive callback and passed by reference to
 * the user's callback. It is only valid for the duration of that call, do not store it.
 *
 * For eager messages, indicated by isRendezvous() returning false, the data is already
 * present and accessible via data() and length(). The handler simply reads the payload and
 * returns; UCXX maps this to UCS_OK.
 *
 * For rendezvous messages, indicated by isRendezvous() returning true, the data is not yet
 * transferred. The handler must provide a buffer and call receive() before returning. If the
 * buffer is allocated
 * from inside the handler, that allocation runs on the worker progress path and blocks
 * progress until it completes. UCXX maps a completed receive() call to UCS_INPROGRESS.
 *
 * The message can be rejected from either path by calling reject(). UCXX forwards the status
 * code to UCX.
 *
 * @note The handler is invoked from a C callback registered with UCX. Exceptions must not
 *       propagate out of the handler. If one escapes, the AM receive callback catches it,
 *       logs a warning, and returns UCS_ERR_IO_ERROR to UCX. Do not rely on exception
 *       propagation for control flow inside the handler.
 */
class AmMessage {
 public:
  /**
   * @brief Pointer to the raw header bytes sent by the remote side.
   *
   * Returns `nullptr` if no header was sent (`headerLength() == 0`).
   */
  const void* header() const noexcept;

  /**
   * @brief Length of the header in bytes.
   */
  size_t headerLength() const noexcept;

  /**
   * @brief Length of the data payload in bytes.
   *
   * Valid for both eager and rendezvous messages.
   */
  size_t length() const noexcept;

  /**
   * @brief Whether this message uses the rendezvous protocol.
   *
   * When `true` the data is not yet available; call `receive()` to pull it.
   * When `false` the data is immediately accessible via `data()`.
   */
  bool isRendezvous() const noexcept;

  /**
   * @brief UCX reply endpoint, or `nullptr` if none was provided by the sender.
   */
  ucp_ep_h replyEp() const noexcept;

  /**
   * @brief Pointer to the eager data payload.
   *
   * Only valid when `!isRendezvous()`. Returns `nullptr` for rendezvous messages to
   * prevent accidental misuse of the UCX data descriptor as application data.
   */
  const void* data() const noexcept;

  /**
   * @brief Initiate an asynchronous rendezvous data pull.
   *
   * Must be called from the handler when `isRendezvous()` is true. The buffer must already
   * be allocated before this call. If the allocation is performed in the handler, it runs on
   * the worker progress path and may block progress until it completes. UCXX returns
   * `UCS_INPROGRESS` after the handler completes when this method has been called.
   *
   * The returned `Request` is registered with the worker's inflight map and will stay alive
   * until the transfer completes. The caller may discard the `shared_ptr` and rely on
   * `callbackFunction` for completion notification instead.
   *
   * @param[in] buffer              pre-allocated receive buffer.
   * @param[in] count               byte count (contig) or IOV segment count.
   * @param[in] datatype            UCP datatype (default: `ucp_dt_make_contig(1)`).
   * @param[in] enablePythonFuture  whether a Python future should be created.
   * @param[in] callbackFunction    called on transfer completion.
   * @param[in] callbackData        data passed to `callbackFunction`.
   *
   * @returns The in-flight receive `Request`.
   */
  std::shared_ptr<Request> receive(void* buffer,
                                   size_t count,
                                   ucp_datatype_t datatype = ucp_dt_make_contig(1),
                                   bool enablePythonFuture = false,
                                   RequestCallbackUserFunction callbackFunction = nullptr,
                                   RequestCallbackUserData callbackData         = nullptr);

  /**
   * @brief Reject the message with the given UCX status code.
   *
   * UCXX forwards `reason` to UCX as the handler return value.
   *
   * @param[in] reason  UCX error status (default: `UCS_ERR_REJECTED`).
   */
  void reject(ucs_status_t reason = UCS_ERR_REJECTED);

 private:
  friend class Worker;

  AmMessage(Worker* worker,
            ucp_ep_h replyEp,
            const void* header,
            size_t headerLength,
            void* data,
            size_t length,
            const ucp_am_recv_param_t* param);

  Worker* _worker;
  ucp_ep_h _replyEp;
  const void* _header;
  size_t _headerLength;
  void* _data;
  size_t _length;
  const ucp_am_recv_param_t* _param;
  bool _receiveCalled{false};
  bool _rejected{false};
  ucs_status_t _rejectStatus{UCS_ERR_REJECTED};
};

}  // namespace ucxx
