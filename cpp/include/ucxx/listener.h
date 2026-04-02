/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/worker.h>

namespace ucxx {

/**
 * @brief Component encapsulating a UCP listener.
 *
 * The UCP layer provides a handle to access listeners in form of `ucp_listener_h` object,
 * this class encapsulates that object and provides methods to simplify its handling.
 */
class Listener : public Component {
 private:
  ucp_listener_h _handle{nullptr};  ///< The UCP listener handle
  std::string _ip{};                ///< The IP address to which the listener is bound to
  uint16_t _port{0};                ///< The port to which the listener is bound to

  /**
   * @brief Private constructor of `ucxx::Listener`.
   *
   * This is the internal implementation of `ucxx::Listener` constructor, made private not
   * to be called directly. Instead the user should call `worker::createListener()` or
   * `ucxx::createListener()`.
   *
   *
   * @param[in] worker        the worker from which to create the listener.
   * @param[in] port          the port which the listener should be bound to.
   * @param[in] callback      user-defined callback to be executed on incoming client
   *                          connections.
   * @param[in] callbackArgs  argument to be passed to the callback.
   */
  Listener(std::shared_ptr<Worker> worker,
           uint16_t port,
           ucp_listener_conn_callback_t callback,
           void* callbackArgs);

 public:
  Listener()                           = delete;
  Listener(const Listener&)            = delete;
  Listener& operator=(Listener const&) = delete;
  Listener(Listener&& o)               = delete;
  Listener& operator=(Listener&& o)    = delete;

  ~Listener();

  /**
   * @brief Constructor of `shared_ptr<ucxx::Listener>`.
   *
   * The constructor for a `shared_ptr<ucxx::Listener>` object. The default constructor is
   * made private to ensure all UCXX objects are shared pointers for correct lifetime
   * management.
   *
   * @code{.cpp}
   *
   * typedef struct ClientContext {
   *   std::shared_ptr<ucxx::Endpoint> endpoint{nullptr};
   *   std::shared_ptr<ucxx::Listener> listener{nullptr};
   * } ClientContextType;
   *
   * void myCallback(ucp_conn_request_h connRequest, void* arg) {
   *   ClientContextType clientContext = (ClientContextType*);
   *   clientContext->endpoint =
   *     clientContext->listener->createEndpointFromConnRequest(connRequest);
   * }
   *
   * ClientContext clientContext;
   *
   * // worker is `std::shared_ptr<ucxx::Worker>`
   * auto listener = worker->createListener(12345, myCallback, clientContext);
   * clientContext->listener = listener;
   *
   * // Equivalent to line above
   * // auto listener = ucxx::createListener(worker, 12345, myCallback, clientContext);
   * @endcode
   *
   * @param[in] worker        the worker from which to create the listener.
   * @param[in] port          the port which the listener should be bound to.
   * @param[in] callback      user-defined callback to be executed on incoming client
   *                          connections.
   * @param[in] callbackArgs  argument to be passed to the callback.
   *
   * @returns The `shared_ptr<ucxx::Listener>` object.
   */
  friend std::shared_ptr<Listener> createListener(std::shared_ptr<Worker> worker,
                                                  uint16_t port,
                                                  ucp_listener_conn_callback_t callback,
                                                  void* callbackArgs);

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
   * @param[in] connRequest           handle to connection request delivered by a
   *                                  listener callback.
   * @param[in] endpointErrorHandling whether to enable endpoint error handling.
   *
   * @returns The `shared_ptr<ucxx::Endpoint>` object.
   */
  [[nodiscard]] std::shared_ptr<Endpoint> createEndpointFromConnRequest(
    ucp_conn_request_h connRequest, bool endpointErrorHandling = true);

  /**
   * @brief Get the underlying `ucp_listener_h` handle.
   *
   * Lifetime of the `ucp_listener_h` handle is managed by the `ucxx::Listener` object and
   * its ownership is non-transferrable. Once the `ucxx::Listener` is destroyed the handle
   * is not valid anymore, it is the user's responsibility to ensure the owner's lifetime
   * while using the handle.
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
