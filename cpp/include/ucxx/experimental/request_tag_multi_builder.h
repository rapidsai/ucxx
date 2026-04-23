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
class RequestTagMulti;

namespace experimental {

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestTagMulti>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::RequestTagMulti>`,
 * allowing optional parameters to be specified via method chaining. Construction happens when
 * the builder expression completes (via implicit conversion) or when `build()` is called.
 *
 * The `endpoint` and `requestData` are required and must be provided to
 * `createRequestTagMulti()`. The `pythonFuture()` method is optional.
 *
 * @code{.cpp}
 *   // Minimal usage (only required args)
 *   auto req = ucxx::experimental::createRequestTagMulti(endpoint, tagMultiSendData).build();
 *
 *   // With optional parameters
 *   auto req = ucxx::experimental::createRequestTagMulti(endpoint, tagMultiSendData)
 *                .pythonFuture(true)
 *                .build();
 *
 *   // Using implicit conversion
 *   std::shared_ptr<ucxx::RequestTagMulti> req =
 *     ucxx::experimental::createRequestTagMulti(endpoint, tagMultiRecvData);
 * @endcode
 */
class RequestTagMultiBuilder {
 private:
  std::shared_ptr<Endpoint> _endpoint;  ///< Parent endpoint (required)
  std::variant<data::TagMultiSend, data::TagMultiReceive>
    _requestData;                       ///< Request-specific data (required)
  bool _enablePythonFuture{false};      ///< Enable Python future support

 public:
  /**
   * @brief Constructor for `RequestTagMultiBuilder` with required parameters.
   *
   * @param[in] endpoint     the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] requestData  container of the specified message type, including all
   *                         type-specific data.
   */
  explicit RequestTagMultiBuilder(
    std::shared_ptr<Endpoint> endpoint,
    std::variant<data::TagMultiSend, data::TagMultiReceive> requestData);

  /**
   * @brief Configure Python future support.
   *
   * @param[in] enable whether a Python future should be created and notified (default: true).
   * @return Reference to this builder for method chaining.
   */
  RequestTagMultiBuilder& pythonFuture(bool enable = true);

  /**
   * @brief Build and return the `RequestTagMulti`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestTagMulti>` object.
   */
  std::shared_ptr<RequestTagMulti> build() const;

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestTagMulti>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestTagMulti>` object.
   */
  operator std::shared_ptr<RequestTagMulti>() const;
};

/**
 * @brief Create a RequestTagMultiBuilder for constructing a
 * `shared_ptr<ucxx::RequestTagMulti>`.
 *
 * @code{.cpp}
 *   auto req = ucxx::experimental::createRequestTagMulti(endpoint, tagMultiSendData)
 *                .pythonFuture(true)
 *                .build();
 * @endcode
 *
 * @param[in] endpoint     the parent endpoint (required).
 * @param[in] requestData  container of the specified message type (required).
 * @return A RequestTagMultiBuilder object that can be used to set optional parameters.
 */
inline RequestTagMultiBuilder createRequestTagMulti(
  std::shared_ptr<Endpoint> endpoint,
  std::variant<data::TagMultiSend, data::TagMultiReceive> requestData)
{
  return RequestTagMultiBuilder(std::move(endpoint), std::move(requestData));
}

}  // namespace experimental

}  // namespace ucxx
