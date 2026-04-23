/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <utility>

#include <ucxx/request_data.h>

namespace ucxx {

// Forward declarations
class Endpoint;
class RequestEndpointClose;

namespace experimental {

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestEndpointClose>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::RequestEndpointClose>`,
 * allowing optional parameters to be specified via method chaining. Construction happens when
 * the builder expression completes (via implicit conversion) or when `build()` is called.
 *
 * The `endpoint` and `requestData` are required and must be provided to
 * `createRequestEndpointClose()`. The remaining methods are optional.
 *
 * @code{.cpp}
 *   // Minimal usage (only required args)
 *   auto req = ucxx::experimental::createRequestEndpointClose(endpoint, closeData).build();
 *
 *   // With optional parameters
 *   auto req = ucxx::experimental::createRequestEndpointClose(endpoint, closeData)
 *                .pythonFuture(true)
 *                .callbackFunction(myCallback)
 *                .callbackData(myData)
 *                .build();
 *
 *   // Using implicit conversion
 *   std::shared_ptr<ucxx::RequestEndpointClose> req =
 *     ucxx::experimental::createRequestEndpointClose(endpoint, closeData);
 * @endcode
 */
class RequestEndpointCloseBuilder {
 private:
  std::shared_ptr<Endpoint> _endpoint;                    ///< Parent endpoint (required)
  data::EndpointClose _requestData;                       ///< Request-specific data (required)
  bool _enablePythonFuture{false};                        ///< Enable Python future support
  RequestCallbackUserFunction _callbackFunction{nullptr}; ///< User callback on completion
  RequestCallbackUserData _callbackData{nullptr};         ///< Data passed to callback

 public:
  /**
   * @brief Constructor for `RequestEndpointCloseBuilder` with required parameters.
   *
   * @param[in] endpoint     the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] requestData  container of the endpoint close request data.
   */
  explicit RequestEndpointCloseBuilder(std::shared_ptr<Endpoint> endpoint,
                                       data::EndpointClose requestData);

  /**
   * @brief Configure Python future support.
   *
   * @param[in] enable whether a Python future should be created and notified (default: true).
   * @return Reference to this builder for method chaining.
   */
  RequestEndpointCloseBuilder& pythonFuture(bool enable = true);

  /**
   * @brief Set the user-defined callback function to call upon completion.
   *
   * @param[in] fn user-defined callback function.
   * @return Reference to this builder for method chaining.
   */
  RequestEndpointCloseBuilder& callbackFunction(RequestCallbackUserFunction fn);

  /**
   * @brief Set the user-defined data to pass to the callback function.
   *
   * @param[in] data user-defined data passed to `callbackFunction`.
   * @return Reference to this builder for method chaining.
   */
  RequestEndpointCloseBuilder& callbackData(RequestCallbackUserData data);

  /**
   * @brief Build and return the `RequestEndpointClose`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestEndpointClose>` object.
   */
  std::shared_ptr<RequestEndpointClose> build() const;

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestEndpointClose>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestEndpointClose>` object.
   */
  operator std::shared_ptr<RequestEndpointClose>() const;
};

/**
 * @brief Create a RequestEndpointCloseBuilder for constructing a
 * `shared_ptr<ucxx::RequestEndpointClose>`.
 *
 * @code{.cpp}
 *   auto req = ucxx::experimental::createRequestEndpointClose(endpoint, closeData)
 *                .pythonFuture(true)
 *                .build();
 * @endcode
 *
 * @param[in] endpoint     the parent endpoint (required).
 * @param[in] requestData  container of the endpoint close request data (required).
 * @return A RequestEndpointCloseBuilder object that can be used to set optional parameters.
 */
inline RequestEndpointCloseBuilder createRequestEndpointClose(
  std::shared_ptr<Endpoint> endpoint, data::EndpointClose requestData)
{
  return RequestEndpointCloseBuilder(std::move(endpoint), std::move(requestData));
}

}  // namespace experimental

}  // namespace ucxx
