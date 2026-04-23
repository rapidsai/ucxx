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
class RequestMem;

namespace experimental {

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestMem>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::RequestMem>`, allowing
 * optional parameters to be specified via method chaining. Construction happens when the
 * builder expression completes (via implicit conversion) or when `build()` is called.
 *
 * The `endpoint` and `requestData` are required and must be provided to
 * `createRequestMem()`. The remaining methods are optional.
 *
 * @code{.cpp}
 *   // Minimal usage (only required args)
 *   auto req = ucxx::experimental::createRequestMem(endpoint, memPutData).build();
 *
 *   // With optional parameters
 *   auto req = ucxx::experimental::createRequestMem(endpoint, memPutData)
 *                .pythonFuture(true)
 *                .callbackFunction(myCallback)
 *                .callbackData(myData)
 *                .build();
 *
 *   // Using implicit conversion
 *   std::shared_ptr<ucxx::RequestMem> req =
 *     ucxx::experimental::createRequestMem(endpoint, memGetData);
 * @endcode
 */
class RequestMemBuilder {
 private:
  std::shared_ptr<Endpoint> _endpoint;                    ///< Parent endpoint (required)
  std::variant<data::MemPut, data::MemGet> _requestData;  ///< Request-specific data (required)
  bool _enablePythonFuture{false};                        ///< Enable Python future support
  RequestCallbackUserFunction _callbackFunction{nullptr}; ///< User callback on completion
  RequestCallbackUserData _callbackData{nullptr};         ///< Data passed to callback

 public:
  /**
   * @brief Constructor for `RequestMemBuilder` with required parameters.
   *
   * @param[in] endpoint     the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] requestData  container of the specified message type, including all
   *                         type-specific data.
   */
  explicit RequestMemBuilder(std::shared_ptr<Endpoint> endpoint,
                             std::variant<data::MemPut, data::MemGet> requestData);

  /**
   * @brief Configure Python future support.
   *
   * @param[in] enable whether a Python future should be created and notified (default: true).
   * @return Reference to this builder for method chaining.
   */
  RequestMemBuilder& pythonFuture(bool enable = true);

  /**
   * @brief Set the user-defined callback function to call upon completion.
   *
   * @param[in] fn user-defined callback function.
   * @return Reference to this builder for method chaining.
   */
  RequestMemBuilder& callbackFunction(RequestCallbackUserFunction fn);

  /**
   * @brief Set the user-defined data to pass to the callback function.
   *
   * @param[in] data user-defined data passed to `callbackFunction`.
   * @return Reference to this builder for method chaining.
   */
  RequestMemBuilder& callbackData(RequestCallbackUserData data);

  /**
   * @brief Build and return the `RequestMem`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestMem>` object.
   */
  std::shared_ptr<RequestMem> build() const;

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestMem>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestMem>` object.
   */
  operator std::shared_ptr<RequestMem>() const;
};

/**
 * @brief Create a RequestMemBuilder for constructing a `shared_ptr<ucxx::RequestMem>`.
 *
 * @code{.cpp}
 *   auto req = ucxx::experimental::createRequestMem(endpoint, memPutData)
 *                .pythonFuture(true)
 *                .build();
 * @endcode
 *
 * @param[in] endpoint     the parent endpoint (required).
 * @param[in] requestData  container of the specified message type (required).
 * @return A RequestMemBuilder object that can be used to set optional parameters.
 */
inline RequestMemBuilder createRequestMem(std::shared_ptr<Endpoint> endpoint,
                                          std::variant<data::MemPut, data::MemGet> requestData)
{
  return RequestMemBuilder(std::move(endpoint), std::move(requestData));
}

}  // namespace experimental

}  // namespace ucxx
