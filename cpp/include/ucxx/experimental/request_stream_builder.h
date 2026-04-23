/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <utility>
#include <variant>

#include <ucxx/request_data.h>

namespace ucxx {

// Forward declarations
class Endpoint;
class RequestStream;

namespace experimental {

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestStream>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::RequestStream>`,
 * allowing optional parameters to be specified via method chaining. Construction happens when
 * the builder expression completes (via implicit conversion) or when `build()` is called.
 *
 * The `endpoint` and `requestData` are required and must be provided to
 * `createRequestStream()`. The `pythonFuture()` method is optional.
 *
 * @code{.cpp}
 *   // Minimal usage (only required args)
 *   auto req = ucxx::experimental::createRequestStream(endpoint, streamSendData).build();
 *
 *   // With optional parameters
 *   auto req = ucxx::experimental::createRequestStream(endpoint, streamSendData)
 *                .pythonFuture(true)
 *                .build();
 *
 *   // Using implicit conversion
 *   std::shared_ptr<ucxx::RequestStream> req =
 *     ucxx::experimental::createRequestStream(endpoint, streamRecvData);
 * @endcode
 */
class RequestStreamBuilder {
 private:
  std::shared_ptr<Endpoint> _endpoint;                              ///< Parent endpoint (required)
  std::variant<data::StreamSend, data::StreamReceive> _requestData; ///< Request-specific data (required)
  bool _enablePythonFuture{false};                                  ///< Enable Python future support

 public:
  /**
   * @brief Constructor for `RequestStreamBuilder` with required parameters.
   *
   * @param[in] endpoint     the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] requestData  container of the specified message type, including all
   *                         type-specific data.
   */
  explicit RequestStreamBuilder(
    std::shared_ptr<Endpoint> endpoint,
    std::variant<data::StreamSend, data::StreamReceive> requestData);

  /**
   * @brief Configure Python future support.
   *
   * @param[in] enable whether a Python future should be created and notified (default: true).
   * @return Reference to this builder for method chaining.
   */
  RequestStreamBuilder& pythonFuture(bool enable = true);

  /**
   * @brief Build and return the `RequestStream`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestStream>` object.
   */
  std::shared_ptr<RequestStream> build() const;

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestStream>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestStream>` object.
   */
  operator std::shared_ptr<RequestStream>() const;
};

/**
 * @brief Create a RequestStreamBuilder for constructing a `shared_ptr<ucxx::RequestStream>`.
 *
 * @code{.cpp}
 *   auto req = ucxx::experimental::createRequestStream(endpoint, streamSendData)
 *                .pythonFuture(true)
 *                .build();
 * @endcode
 *
 * @param[in] endpoint     the parent endpoint (required).
 * @param[in] requestData  container of the specified message type (required).
 * @return A RequestStreamBuilder object that can be used to set optional parameters.
 */
inline RequestStreamBuilder createRequestStream(
  std::shared_ptr<Endpoint> endpoint,
  std::variant<data::StreamSend, data::StreamReceive> requestData)
{
  return RequestStreamBuilder(std::move(endpoint), std::move(requestData));
}

}  // namespace experimental

}  // namespace ucxx
