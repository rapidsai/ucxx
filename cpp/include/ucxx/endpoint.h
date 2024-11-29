/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <netdb.h>

#include <memory>
#include <string>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/address.h>
#include <ucxx/component.h>
#include <ucxx/exception.h>
#include <ucxx/inflight_requests.h>
#include <ucxx/listener.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/worker.h>

namespace ucxx {

/**
 * @brief Deleter for a endpoint parameters object.
 *
 * Deleter used during allocation of a `ucp_ep_params_t*` to handle automated deletion of
 * the object when its reference count goes to zero.
 */
struct EpParamsDeleter {
  /**
   * @brief Execute the deletion.
   *
   * Execute the deletion of the `ucp_ep_params_t*` object.
   *
   * param[in] ptr  the point to the object to be deleted.
   */
  void operator()(ucp_ep_params_t* ptr);
};

/**
 * @brief Component encapsulating a UCP endpoint.
 *
 * The UCP layer provides a handle to access endpoints in form of `ucp_ep_h` object,
 * this class encapsulates that object and provides methods to simplify its handling.
 */
class Endpoint : public Component {
 private:
  ucp_ep_h _handle{nullptr};          ///< Handle to the UCP endpoint
  ucp_ep_h _originalHandle{nullptr};  ///< Handle to the UCP endpoint, after it was previously
                                      ///< closed, used for logging purposes only
  bool _endpointErrorHandling{true};  ///< Whether the endpoint enables error handling
  std::unique_ptr<InflightRequests> _inflightRequests{
    std::make_unique<InflightRequests>()};  ///< The inflight requests
  std::mutex _mutex{std::mutex()};  ///< Mutex used during close to prevent race conditions between
                                    ///< application thread and `ucxx::Endpoint::setCloseCallback()`
                                    ///< that may run asynchronously on another thread.
  ucs_status_t _status{UCS_INPROGRESS};  ///< Endpoint status
  std::atomic<bool> _closing{false};     ///< Prevent calling close multiple concurrent times.
  EndpointCloseCallbackUserFunction _closeCallback{nullptr};  ///< Close callback to call
  EndpointCloseCallbackUserData _closeCallbackArg{
    nullptr};  ///< Argument to be passed to close callback

  /**
   * @brief Private constructor of `ucxx::Endpoint`.
   *
   * This is the internal implementation of `ucxx::Endpoint` constructor, made private not
   * to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * This constructor does not fully initialize the `ucxx::Endpoint` object, the caller
   * must call the `create()` method immediately after this to complete the construction.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Listener::createEndpointFromConnRequest`
   * - `ucxx::Worker::createEndpointFromHostname()`
   * - `ucxx::Worker::createEndpointFromWorkerAddress()`
   * - `ucxx::createEndpointFromConnRequest()`
   * - `ucxx::createEndpointFromHostname()`
   * - `ucxx::createEndpointFromWorkerAddress()`
   *
   * @param[in] workerOrListener      the parent component, which may either be a
   *                                  `std::shared_ptr<Listener>` or
   *                                  `std::shared_ptr<Worker>`.
   * @param[in] endpointErrorHandling whether to enable endpoint error handling.
   */
  Endpoint(std::shared_ptr<Component> workerOrListener, bool endpointErrorHandling);

  /**
   * @brief Create the underlying UCP endpoint of `ucxx::Endpoint`.
   *
   * This is the internal implementation of `ucxx::Endpoint` creation. This method completes
   * the initialization with the creation of the UCP endpoint and must be always called
   * after the private constructor is called.
   *
   * @param[in] params                parameters specifying UCP endpoint capabilities.
   */
  void create(ucp_ep_params_t* params);

  /**
   * @brief Register an inflight request.
   *
   * Called each time a new transfer request is made by the `Endpoint`, such that it may
   * be canceled when necessary. Also schedule requests to be canceled immediately after
   * registration if the endpoint error handler has been called with an error.
   *
   * @param[in] request the request to register.
   *
   * @return the request that was registered (i.e., the `request` argument itself).
   */
  [[nodiscard]] std::shared_ptr<Request> registerInflightRequest(std::shared_ptr<Request> request);

  /**
   * @brief The error callback registered at endpoint creation time.
   *
   * When the endpoint is created with error handling support this method is registered as
   * the callback to be called when the endpoint is closing, it is responsible for checking
   * the closing status and update internal state accordingly. If error handling support is
   * not active, this method is not registered nor called.
   *
   * The signature for this method must match `ucp_err_handler_cb_t`.
   */
  friend void endpointErrorCallback(void* arg, ucp_ep_h ep, ucs_status_t status);

 public:
  Endpoint()                           = delete;
  Endpoint(const Endpoint&)            = delete;
  Endpoint& operator=(Endpoint const&) = delete;
  Endpoint(Endpoint&& o)               = delete;
  Endpoint& operator=(Endpoint&& o)    = delete;

  ~Endpoint();

  /**
   * @brief Constructor for `shared_ptr<ucxx::Endpoint>`.
   *
   * The constructor for a `shared_ptr<ucxx::Endpoint>` object, connecting to a listener
   * from the given hostname or IP address and port pair.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`, with a presumed listener on
   * // "localhost:12345"
   * auto endpoint = worker->createEndpointFromHostname("localhost", 12345, true);
   *
   * // Equivalent to line above
   * // auto endpoint = ucxx::createEndpointFromHostname(worker, "localhost", 12345, true);
   * @endcode
   *
   * @param[in] worker                parent worker from which to create the endpoint.
   * @param[in] ipAddress             hostname or IP address the listener is bound to.
   * @param[in] port                  port the listener is bound to.
   * @param[in] endpointErrorHandling whether to enable endpoint error handling.
   *
   * @returns The `shared_ptr<ucxx::Endpoint>` object
   */
  friend std::shared_ptr<Endpoint> createEndpointFromHostname(std::shared_ptr<Worker> worker,
                                                              std::string ipAddress,
                                                              uint16_t port,
                                                              bool endpointErrorHandling);

  /**
   * @brief Constructor for `shared_ptr<ucxx::Endpoint>`.
   *
   * The constructor for a `shared_ptr<ucxx::Endpoint>` object from a `ucp_conn_request_h`,
   * as delivered by a `ucxx::Listener` connection callback.
   *
   * @code{.cpp}
   * // listener is `std::shared_ptr<ucxx::Listener>`, with a `ucp_conn_request_h` delivered
   * // by a `ucxx::Listener` connection callback.
   * auto endpoint = listener->createEndpointFromConnRequest(connRequest, true);
   *
   * // Equivalent to line above
   * // auto endpoint = ucxx::createEndpointFromConnRequest(listener, connRequest, true);
   * @endcode
   *
   * @param[in] listener              listener from which to create the endpoint.
   * @param[in] connRequest           handle to connection request delivered by a
   *                                  listener callback.
   * @param[in] endpointErrorHandling whether to enable endpoint error handling.
   *
   * @returns The `shared_ptr<ucxx::Endpoint>` object
   */
  friend std::shared_ptr<Endpoint> createEndpointFromConnRequest(std::shared_ptr<Listener> listener,
                                                                 ucp_conn_request_h connRequest,
                                                                 bool endpointErrorHandling);

  /**
   * @brief Constructor for `shared_ptr<ucxx::Endpoint>`.
   *
   * The constructor for a `shared_ptr<ucxx::Endpoint>` object from a `shared_ptr<ucxx::Address>`.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`, address is `std::shared_ptr<ucxx::Address>`
   * auto endpoint = worker->createEndpointFromWorkerAddress(address, true);
   *
   * // Equivalent to line above
   * // auto endpoint = ucxx::createEndpointFromWorkerAddress(worker, address, true);
   * @endcode
   *
   * @param[in] worker                parent worker from which to create the endpoint.
   * @param[in] address               address of the remote UCX worker
   * @param[in] endpointErrorHandling whether to enable endpoint error handling.
   *
   * @returns The `shared_ptr<ucxx::Endpoint>` object
   */
  friend std::shared_ptr<Endpoint> createEndpointFromWorkerAddress(std::shared_ptr<Worker> worker,
                                                                   std::shared_ptr<Address> address,
                                                                   bool endpointErrorHandling);

  /**
   * @brief Get the underlying `ucp_ep_h` handle.
   *
   * Lifetime of the `ucp_ep_h` handle is managed by the `ucxx::Endpoint` object and its
   * ownership is non-transferrable. Once the `ucxx::Endpoint` is destroyed the handle
   * is not valid anymore, it is the user's responsibility to ensure the owner's lifetime
   * while using the handle.
   *
   * @code{.cpp}
   * // endpoint is `std::shared_ptr<ucxx::Endpoint>`
   * ucp_ep_h endpointHandle = endpoint->getHandle();
   * @endcode
   *
   * @returns The underlying `ucp_ep_h` handle.
   */
  [[nodiscard]] ucp_ep_h getHandle();

  /**
   * @brief Check whether the endpoint is still alive.
   *
   * Check whether the endpoint is still alive, generally `true` until `closeBlocking()` is
   * called, `close()` is called and the returned request completes or the endpoint errors
   * and the error handling procedure is executed. Always `true` if endpoint error handling
   * is disabled.
   *
   * @returns whether the endpoint is still alive if endpoint enables error handling, always
   *          returns `true` if error handling is disabled.
   */
  [[nodiscard]] bool isAlive() const;

  /**
   * @brief Raises an exception if an error occurred.
   *
   * Raises an exception if an error occurred and error handling is enabled for the
   * endpoint, no-op otherwise.
   *
   * @throws ucxx::ConnectionResetError if `UCP_ERR_CONNECTION_RESET` occurred.
   * @throws ucxx::Error                if any other UCP error occurred.
   */
  void raiseOnError();

  /**
   * @brief Remove reference to request from internal container.
   *
   * Remove the reference to a specific request from the internal container. This should
   * be called when a request has completed and the `ucxx::Endpoint` does not need to keep
   * track of it anymore. The raw pointer to a `ucxx::Request` is passed here as opposed
   * to the usual `std::shared_ptr<ucxx::Request>` used elsewhere, this is because the
   * raw pointer address is used as key to the requests reference, and this is called
   * from the object's destructor.
   *
   * @param[in] request raw pointer to the request
   */
  void removeInflightRequest(const Request* const request);

  /**
   * @brief Cancel inflight requests.
   *
   * Cancel inflight requests, returning the total number of requests that were scheduled
   * for cancelation. After the requests are scheduled for cancelation, the caller must
   * progress the worker and check the result of `getCancelingSize()`, all requests are only
   * canceled when `getCancelingSize()` returns `0`.
   *
   * @returns Number of requests that were scheduled for cancelation.
   */
  size_t cancelInflightRequests();

  /**
   * @brief Check the number of inflight requests being canceled.
   *
   * Check the number of inflight requests that were scheduled for cancelation with
   * `cancelInflightRequests()` who have not yet completed cancelation. To ensure their
   * cancelation is completed, the worker must be progressed until this method returns `0`.
   *
   * @returns Number of requests that are in process of cancelation.
   */
  [[nodiscard]] size_t getCancelingSize() const;

  /**
   * @brief Cancel inflight requests.
   *
   * Cancel inflight requests and block until all requests complete cancelation, returning
   * the total number of requests that were canceled.  This is usually executed by
   * `closeBlocking()`, when pending requests will no longer be able to complete.
   *
   * If the parent worker is running a progress thread, a maximum timeout may be specified
   * for which the close operation will wait. This can be particularly important for cases
   * where the progress thread might be attempting to acquire a resource (e.g., the Python
   * GIL) while the current thread owns that resource. In particular for Python, the
   * `~Endpoint()` will call this method for which we can't release the GIL when the garbage
   * collector runs and destroys the object.
   *
   * @param[in] period      maximum period to wait for a generic pre/post progress thread
   *                        operation will wait for.
   * @param[in] maxAttempts maximum number of attempts to close endpoint, only applicable
   *                         if worker is running a progress thread and `period > 0`.
   *
   * @returns Number of requests that were canceled.
   */
  size_t cancelInflightRequestsBlocking(uint64_t period = 0, uint64_t maxAttempts = 1);

  /**
   * @brief Register a user-defined callback to call when endpoint closes.
   *
   * Register a user-defined callback and argument that is later called immediately after
   * the endpoint closes. The callback is executed either if the endpoint closed
   * successfully after completing and disconnecting from the remote endpoint, but more
   * importantly when any error occurs, allowing the application to be notified immediately
   * after such an event occurred.
   *
   * @throws  std::runtime_error  if the endpoint is closing or has already closed and this
   *                              is not removing the close callback (setting both
   *                              `closeCallback` and `closeCallbackArg` to `nullptr`)
   *
   * @param[in] closeCallback     `std::function` to a function definition return `void` and
   *                              receiving a single opaque pointer.
   * @param[in] closeCallbackArg  pointer to optional user-allocated callback argument.
   */
  void setCloseCallback(EndpointCloseCallbackUserFunction closeCallback,
                        EndpointCloseCallbackUserData closeCallbackArg);

  /**
   * @brief Enqueue an active message send operation.
   *
   * Enqueue an active message send operation, returning a `std::shared_ptr<ucxx::Request>`
   * that can be later awaited and checked for errors. This is a non-blocking operation, and
   * the status of the transfer must be verified from the resulting request object before
   * the data can be released.
   *
   * An optional `receiverCallbackInfo` may be specified, in which case the remote worker
   * obligatorily needs to have registered a callback with the same `receiverCallbackInfo`
   * in order to execute the callback when the active message is received. When this is
   * specified, `amRecv()` will _NOT_ match this message, which is instead handled by the
   * remote worker's callback.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] buffer                  a raw pointer to the data to be sent.
   * @param[in] length                  the size in bytes of the tag message to be sent.
   * @param[in] memoryType              the memory type of the buffer.
   * @param[in] receiverCallbackInfo    the owner name and unique identifier of the receiver
                                        callback.
   * @param[in] enablePythonFuture      whether a python future should be created and
   *                                    subsequently notified.
   * @param[in] callbackFunction        user-defined callback function to call upon
                                        completion.
   * @param[in] callbackData            user-defined data to pass to the `callbackFunction`.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> amSend(
    void* buffer,
    const size_t length,
    const ucs_memory_type_t memoryType,
    const std::optional<AmReceiverCallbackInfo> receiverCallbackInfo = std::nullopt,
    const bool enablePythonFuture                                    = false,
    RequestCallbackUserFunction callbackFunction                     = nullptr,
    RequestCallbackUserData callbackData                             = nullptr);

  /**
   * @brief Enqueue an active message receive operation.
   *
   * Enqueue an active message receive operation, returning a
   * `std::shared_ptr<ucxx::Request>` that can be later awaited and checked for errors,
   * making data available via the return value's `getRecvBuffer()` method once the
   * operation completes successfully. This is a non-blocking operation, and the status of
   * the transfer must be verified from the resulting request object before the data can be
   * consumed.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   *
   * @returns Request to be subsequently checked for the completion state and data.
   */
  [[nodiscard]] std::shared_ptr<Request> amRecv(
    const bool enablePythonFuture                = false,
    RequestCallbackUserFunction callbackFunction = nullptr,
    RequestCallbackUserData callbackData         = nullptr);

  /**
   * @brief Enqueue a memory put operation.
   *
   * Enqueue a memory operation, returning a `std::shared<ucxx::Request>` that can be later
   * awaited and checked for errors. This is a non-blocking operation, and the status of the
   * transfer must be verified from the resulting request object before both local and
   * remote data can be released and the remote data can be consumed.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the tag message to be sent.
   * @param[in] remoteAddr          the destination remote memory address to write to.
   * @param[in] rkey                the remote memory key associated with the remote memory
   *                                address.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> memPut(
    void* buffer,
    size_t length,
    uint64_t remote_addr,
    ucp_rkey_h rkey,
    const bool enablePythonFuture                = false,
    RequestCallbackUserFunction callbackFunction = nullptr,
    RequestCallbackUserData callbackData         = nullptr);

  /**
   * @brief Enqueue a memory put operation.
   *
   * Enqueue a memory operation, returning a `std::shared<ucxx::Request>` that can be later
   * awaited and checked for errors. This is a non-blocking operation, and the status of the
   * transfer must be verified from the resulting request object before both local and
   * remote data can be released and the remote data can be consumed.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the tag message to be sent.
   * @param[in] remoteKey           the remote memory key associated with the remote memory
   *                                address.
   * @param[in] remoteAddrOffset    the destination remote memory address offset where to
   *                                start writing to, `0` means start writing from beginning
   *                                of the base address.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> memPut(
    void* buffer,
    size_t length,
    std::shared_ptr<ucxx::RemoteKey> remoteKey,
    uint64_t remoteAddrOffset                    = 0,
    const bool enablePythonFuture                = false,
    RequestCallbackUserFunction callbackFunction = nullptr,
    RequestCallbackUserData callbackData         = nullptr);

  /**
   * @brief Enqueue a memory get operation.
   *
   * Enqueue a memory operation, returning a `std::shared<ucxx::Request>` that can be later
   * awaited and checked for errors. This is a non-blocking operation, and the status of the
   * transfer must be verified from the resulting request object before both local and
   * remote data can be released and the local data can be consumed.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the tag message to be sent.
   * @param[in] remoteAddr          the source remote memory address to read from.
   * @param[in] rkey                the remote memory key associated with the remote memory
   *                                address.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> memGet(
    void* buffer,
    size_t length,
    uint64_t remoteAddr,
    ucp_rkey_h rkey,
    const bool enablePythonFuture                = false,
    RequestCallbackUserFunction callbackFunction = nullptr,
    RequestCallbackUserData callbackData         = nullptr);

  /**
   * @brief Enqueue a memory get operation.
   *
   * Enqueue a memory operation, returning a `std::shared<ucxx::Request>` that can be later
   * awaited and checked for errors. This is a non-blocking operation, and the status of the
   * transfer must be verified from the resulting request object before both local and
   * remote data can be released and the local data can be consumed.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the tag message to be sent.
   * @param[in] remoteKey           the remote memory key associated with the remote memory
   *                                address.
   * @param[in] remoteAddrOffset    the destination remote memory address offset where to
   *                                start reading from, `0` means start writing from
   *                                beginning of the base address.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> memGet(
    void* buffer,
    size_t length,
    std::shared_ptr<ucxx::RemoteKey> remoteKey,
    uint64_t remoteAddrOffset                    = 0,
    const bool enablePythonFuture                = false,
    RequestCallbackUserFunction callbackFunction = nullptr,
    RequestCallbackUserData callbackData         = nullptr);

  /**
   * @brief Enqueue a stream send operation.
   *
   * Enqueue a stream send operation, returning a `std::shared<ucxx::Request>` that can
   * be later awaited and checked for errors. This is a non-blocking operation, and the
   * status of the transfer must be verified from the resulting request object before the
   * data can be released.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the tag message to be sent.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> streamSend(void* buffer,
                                                    size_t length,
                                                    const bool enablePythonFuture);

  /**
   * @brief Enqueue a stream receive operation.
   *
   * Enqueue a stream receive operation, returning a `std::shared<ucxx::Request>` that can
   * be later awaited and checked for errors. This is a non-blocking operation, and the
   * status of the transfer must be verified from the resulting request object before the
   * data can be consumed.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] buffer              a raw pointer to pre-allocated memory where resulting
   *                                data will be stored.
   * @param[in] length              the size in bytes of the tag message to be received.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> streamRecv(void* buffer,
                                                    size_t length,
                                                    const bool enablePythonFuture);

  /**
   * @brief Enqueue a tag send operation.
   *
   * Enqueue a tag send operation, returning a `std::shared<ucxx::Request>` that can
   * be later awaited and checked for errors. This is a non-blocking operation, and the
   * status of the transfer must be verified from the resulting request object before the
   * data can be released.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the tag message to be sent.
   * @param[in] tag                 the tag to match.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> tagSend(
    void* buffer,
    size_t length,
    Tag tag,
    const bool enablePythonFuture                = false,
    RequestCallbackUserFunction callbackFunction = nullptr,
    RequestCallbackUserData callbackData         = nullptr);

  /**
   * @brief Enqueue a tag receive operation.
   *
   * Enqueue a tag receive operation, returning a `std::shared<ucxx::Request>` that can
   * be later awaited and checked for errors. This is a non-blocking operation, and the
   * status of the transfer must be verified from the resulting request object before the
   * data can be consumed.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] buffer              a raw pointer to pre-allocated memory where resulting
   *                                data will be stored.
   * @param[in] length              the size in bytes of the tag message to be received.
   * @param[in] tag                 the tag to match.
   * @param[in] tagMask             the tag mask to use.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> tagRecv(
    void* buffer,
    size_t length,
    Tag tag,
    TagMask tagMask,
    const bool enablePythonFuture                = false,
    RequestCallbackUserFunction callbackFunction = nullptr,
    RequestCallbackUserData callbackData         = nullptr);

  /**
   * @brief Enqueue a multi-buffer tag send operation.
   *
   * Enqueue a multi-buffer tag send operation, returning a
   * `std::shared<ucxx::RequestTagMulti>` that can be later awaited and checked for errors.
   * This is a non-blocking operation, and the status of the transfer must be verified from
   * the resulting request object before the data can be released.
   *
   * The primary use of multi-buffer transfers is in Python where we want to reduce the
   * amount of futures needed to watch for, thus reducing Python overhead. However, this
   * may be used as a convenience implementation for transfers that require multiple
   * frames, internally this is implemented as one or more `tagSend` calls sending headers
   * (depending on the number of frames being transferred), followed by one `tagSend` for
   * each data frame.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @throws  std::runtime_error  if sizes of `buffer`, `size` and `isCUDA` do not match.
   *
   * @param[in] buffer              a vector of raw pointers to the data frames to be sent.
   * @param[in] size                a vector of size in bytes of each frame to be sent.
   * @param[in] isCUDA              a vector of booleans (integers to prevent incoherence
   *                                with other vector types) indicating whether frame is
   *                                CUDA, to ensure proper memory allocation by the
   *                                receiver.
   * @param[in] tag                 the tag to match.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> tagMultiSend(const std::vector<void*>& buffer,
                                                      const std::vector<size_t>& size,
                                                      const std::vector<int>& isCUDA,
                                                      const Tag tag,
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
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] tag                 the tag to match.
   * @param[in] tagMask             the tag mask to use.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> tagMultiRecv(const Tag tag,
                                                      const TagMask tagMask,
                                                      const bool enablePythonFuture);

  /**
   * @brief Enqueue a flush operation.
   *
   * Enqueue request to flush outstanding AMO (Atomic Memory Operation) and RMA (Remote
   * Memory Access) operations on the endpoint, returning a pointer to a request object that
   * can be later awaited and checked for errors. This is a non-blocking operation, and its
   * status must be verified from the resulting request object to confirm the flush
   * operation has completed successfully.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the transfer has completed. Requires UCXX Python support.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   *
   * @returns Request to be subsequently checked for the completion and its state.
   */
  [[nodiscard]] std::shared_ptr<Request> flush(
    const bool enablePythonFuture                = false,
    RequestCallbackUserFunction callbackFunction = nullptr,
    RequestCallbackUserData callbackData         = nullptr);

  /**
   * @brief Get `ucxx::Worker` component from a worker or listener object.
   *
   * A `std::shared_ptr<ucxx::Endpoint>` needs to be created and registered by
   * `std::shared_ptr<ucxx::Worker>`, but the endpoint may be a child of a
   * `std::shared_ptr<ucxx::Listener>` object. For convenience, this method can be used to
   * get the `std::shared_ptr<ucxx::Worker>` which the endpoint is associated with.
   *
   * @returns The `std::shared_ptr<ucxx::Worker>` which the endpoint is associated with.
   */
  [[nodiscard]] std::shared_ptr<Worker> getWorker();

  /**
   * @brief Enqueue a non-blocking endpoint close operation.
   *
   * Enqueue a non-blocking endpoint close operation, which will close the endpoint without
   * requiring to destroy the object. This may be useful when other
   * `std::shared_ptr<ucxx::Request>` objects are still alive, such as inflight transfers.
   *
   * This method returns a `std::shared<ucxx::Request>` that can be later awaited and
   * checked for errors. This is a non-blocking operation, and the status of closing the
   * endpoint must be verified from the resulting request object before the
   * `std::shared_ptr<ucxx::Endpoint>` can be safely destroyed and the UCP endpoint assumed
   * inactive (closed). If the endpoint is already closed or in process of closing, `nullptr`
   * is returned instead.
   *
   * If the endpoint was created with error handling support, the error callback will be
   * executed, implying the user-defined callback will also be executed.
   *
   * If a user-defined callback is specified via the `callbackFunction` argument then that
   * callback will be executed, if not then the callback registered with `setCloseCallback()`
   * will be executed, if neither was specified then no user-defined callback will be
   * executed.
   *
   * Using a Python future may be requested by specifying `enablePythonFuture`. If a
   * Python future is requested, the Python application must then await on this future to
   * ensure the close operation has completed. Requires UCXX Python support.
   *
   * @warning Unlike its `closeBlocking()` counterpart, this method does not cancel any
   * inflight requests prior to submitting the UCP close request. Before scheduling the
   * endpoint close request, the caller must first call `cancelInflightRequests()` and
   * progress the worker until `getCancelingSize()` returns `0`.
   *
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   *
   * @returns Request to be subsequently checked for the completion and its state, or
   *          `nullptr` if the endpoint has already closed or is already in process of
   *          closing.
   */
  [[nodiscard]] std::shared_ptr<Request> close(
    const bool enablePythonFuture                      = false,
    EndpointCloseCallbackUserFunction callbackFunction = nullptr,
    EndpointCloseCallbackUserData callbackData         = nullptr);

  /**
   * @brief Close the endpoint while keeping the object alive.
   *
   * Close the endpoint without requiring to destroy the object, blocking until the
   * operation completes. This may be useful when `std::shared_ptr<ucxx::Request>` objects
   * are still alive.
   *
   * If the endpoint was created with error handling support, the error callback will be
   * executed, implying the user-defined callback will also be executed if one was
   * registered with `setCloseCallback()`.
   *
   * If the parent worker is running a progress thread, a maximum timeout may be specified
   * for which the close operation will wait. This can be particularly important for cases
   * where the progress thread might be attempting to acquire a resource (e.g., the Python
   * GIL) while the current thread owns that resource. In particular for Python, the
   * `~Endpoint()` will call this method for which we can't release the GIL when the garbage
   * collector runs and destroys the object.
   *
   * @param[in] period      maximum period to wait for a generic pre/post progress thread
   *                        operation will wait for.
   * @param[in] maxAttempts maximum number of attempts to close endpoint, only applicable
   *                        if worker is running a progress thread and `period > 0`.
   */
  void closeBlocking(uint64_t period = 0, uint64_t maxAttempts = 1);
};

}  // namespace ucxx
