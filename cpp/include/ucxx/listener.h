/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint_builder.h>
#include <ucxx/listener_builder.h>
#include <ucxx/worker.h>

namespace ucxx {

class Listener;
/**
 * @brief Component encapsulating a UCP listener.
 *
 * The UCP layer provides a handle to access listeners in form of `ucp_listener_h` object,
 * this class encapsulates that object and provides methods to simplify its handling.
 */
class Listener : public Component {
 private:
  ucp_listener_h _handle{nullptr};                  ///< The UCP listener handle
  std::mutex _handleMutex{};                        ///< Mutex protecting `_handle`
  std::atomic<bool> _closing{false};                ///< Whether `close()` was called
  ucp_listener_conn_callback_t _callback{nullptr};  ///< User-defined connection callback
  void* _callbackArgs{nullptr};                     ///< Argument to `_callback`
  std::string _ip{};  ///< The IP address to which the listener is bound to
  uint16_t _port{0};  ///< The port to which the listener is bound to

  /**
   * @brief Private constructor of `ucxx::Listener`.
   *
   * This is the internal implementation of `ucxx::Listener` constructor, made private not
   * to be called directly.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Worker::listenerBuilder()`
   * - `ucxx::ListenerBuilder`
   *
   *
   * @param[in] worker        the worker from which to create the listener.
   * @param[in] host          the hostname or IP address which the listener should be
   *                          bound to, an empty string binds to all interfaces
   *                          (wildcard).
   * @param[in] port          the port which the listener should be bound to.
   * @param[in] callback      user-defined callback to be executed on incoming client
   *                          connections.
   * @param[in] callbackArgs  argument to be passed to the callback.
   */
  Listener(std::shared_ptr<Worker> worker,
           std::string host,
           uint16_t port,
           ucp_listener_conn_callback_t callback,
           void* callbackArgs);

  /**
   * @brief Connection handler registered with UCP.
   *
   * Reject connection requests once `close()` was called, without calling the
   * user-defined callback, otherwise call the user-defined callback.
   *
   * @param[in] connRequest handle to the connection request.
   * @param[in] arg         pointer to the `ucxx::Listener` object.
   */
  static void connCallback(ucp_conn_request_h connRequest, void* arg);

  /**
   * @brief Destroy the underlying `ucp_listener_h`, if not yet destroyed.
   */
  void destroyHandle();

 public:
  Listener()                           = delete;
  Listener(const Listener&)            = delete;
  Listener& operator=(Listener const&) = delete;
  Listener(Listener&& o)               = delete;
  Listener& operator=(Listener&& o)    = delete;

  ~Listener();

  // Allow internal construction without exposing factory functions.
  friend class detail::ConstructorFactory;

  /**
   * @brief Create a builder for an endpoint from a connection request.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion creates the endpoint from this listener.
   *
   * @param[in] connRequest handle to connection request delivered by a listener callback.
   * @returns Builder to configure optional endpoint parameters.
   */
  [[nodiscard]] EndpointBuilder endpointBuilder(ucp_conn_request_h connRequest);

  /**
   * @brief Close the listener, destroying the underlying `ucp_listener_h`.
   *
   * Stop accepting new connections by destroying the underlying UCP listener. Once this
   * method returns, the listener callback is not called anymore, including for
   * connection requests that were already received but not yet delivered, and it is
   * thus safe to release any resources referenced by the callback argument. Endpoints
   * previously created from this listener remain valid and may continue to keep this
   * object alive. Connection requests that are delivered after `close()` was called are
   * rejected without calling the listener callback, a listener callback that was already
   * executing may still use the listener, e.g., to create an endpoint from its
   * connection request.
   *
   * This method is idempotent and is called by the destructor. Every call returns only
   * after the UCP listener was destroyed, unless the timeout described below expires.
   *
   * When the worker progress thread is running, this method may be called from any
   * thread. Called from the progress thread, the UCP listener is destroyed immediately.
   * Called from any other thread, it blocks until the progress thread has destroyed the
   * UCP listener (or a 10 second timeout expires), so the caller must not hold resources
   * the progress thread may need to make progress (e.g., the Python GIL). Without a
   * progress thread, the UCP listener is destroyed on the calling thread, which must be
   * the thread that progresses the worker.
   *
   * Calling this method from within this listener's own callback is supported in all
   * progress modes. Calling it from within any other callback executed during worker
   * progress (e.g., endpoint or request callbacks) is not supported, and the same applies
   * to releasing the last reference to the listener there, since the destructor calls
   * this method.
   *
   * If the progress thread does not destroy the UCP listener within 10 seconds an error
   * is logged and `isClosed()` remains `false`, in that case the callback argument must
   * not be released. Connection requests delivered afterwards are still rejected, and
   * the destructor retries destroying the UCP listener.
   *
   * @code{.cpp}
   * // listener is `std::shared_ptr<ucxx::Listener>`
   * listener->close();
   * assert(listener->isClosed());
   * @endcode
   */
  void close();

  /**
   * @brief Reject a connection request.
   *
   * Reject a connection request delivered to the listener callback, the client is
   * notified the connection was rejected. Must be called before the listener is closed,
   * for example from the listener callback.
   *
   * @param[in] connRequest handle to connection request delivered by a listener callback.
   */
  void reject(ucp_conn_request_h connRequest);

  /**
   * @brief Check whether the listener has been closed.
   *
   * @returns `true` if the underlying `ucp_listener_h` was destroyed by `close()`,
   *          `false` otherwise.
   */
  [[nodiscard]] bool isClosed();

  /**
   * @brief Get the underlying `ucp_listener_h` handle.
   *
   * Lifetime of the `ucp_listener_h` handle is managed by the `ucxx::Listener` object and
   * its ownership is non-transferrable. Once the `ucxx::Listener` is destroyed the handle
   * is not valid anymore, it is the user's responsibility to ensure the owner's lifetime
   * while using the handle. After `close()` is called, `nullptr` is returned.
   *
   * @code{.cpp}
   * // listener is `std::shared_ptr<ucxx::Listener>`
   * ucp_listener_h listenerHandle = listener->getHandle();
   * @endcode
   *
   * @returns The underlying `ucp_listener_h` handle.
   */
  [[nodiscard]] ucp_listener_h getHandle();

  /**
   * @brief Get the port to which the listener is bound to.
   *
   * Get the port to which the listener is bound to.
   *
   * @returns the port to which the listener is bound to.
   */
  [[nodiscard]] uint16_t getPort();

  /**
   * @brief Get the IP address to which the listener is bound to.
   *
   * Get the IP address to which the listener is bound to.
   *
   * @returns the IP address to which the listener is bound to.
   */
  [[nodiscard]] std::string getIp();
};

}  // namespace ucxx
