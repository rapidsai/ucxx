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
class Component;
class RequestFlush;

namespace experimental {

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestFlush>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::RequestFlush>`, allowing
 * optional parameters to be specified via method chaining. Construction happens when the
 * builder expression completes (via implicit conversion) or when `build()` is called.
 *
 * The `endpointOrWorker` and `requestData` are required and must be provided to
 * `createRequestFlush()`. The remaining methods are optional.
 *
 * @code{.cpp}
 *   // Minimal usage (only required args)
 *   auto req = ucxx::experimental::createRequestFlush(endpointOrWorker, flushData).build();
 *
 *   // With optional parameters
 *   auto req = ucxx::experimental::createRequestFlush(endpointOrWorker, flushData)
 *                .pythonFuture(true)
 *                .callbackFunction(myCallback)
 *                .callbackData(myData)
 *                .build();
 *
 *   // Using implicit conversion
 *   std::shared_ptr<ucxx::RequestFlush> req =
 *     ucxx::experimental::createRequestFlush(endpointOrWorker, flushData);
 * @endcode
 */
class RequestFlushBuilder {
 private:
  std::shared_ptr<Component> _endpointOrWorker;           ///< Parent endpoint or worker (required)
  data::Flush _requestData;                               ///< Request-specific data (required)
  bool _enablePythonFuture{false};                        ///< Enable Python future support
  RequestCallbackUserFunction _callbackFunction{nullptr}; ///< User callback on completion
  RequestCallbackUserData _callbackData{nullptr};         ///< Data passed to callback

 public:
  /**
   * @brief Constructor for `RequestFlushBuilder` with required parameters.
   *
   * @param[in] endpointOrWorker  the parent component, which may be a
   *                              `std::shared_ptr<Endpoint>` or `std::shared_ptr<Worker>`.
   * @param[in] requestData       container of the flush request data.
   */
  explicit RequestFlushBuilder(std::shared_ptr<Component> endpointOrWorker,
                               data::Flush requestData);

  /**
   * @brief Configure Python future support.
   *
   * @param[in] enable whether a Python future should be created and notified (default: true).
   * @return Reference to this builder for method chaining.
   */
  RequestFlushBuilder& pythonFuture(bool enable = true);

  /**
   * @brief Set the user-defined callback function to call upon completion.
   *
   * @param[in] fn user-defined callback function.
   * @return Reference to this builder for method chaining.
   */
  RequestFlushBuilder& callbackFunction(RequestCallbackUserFunction fn);

  /**
   * @brief Set the user-defined data to pass to the callback function.
   *
   * @param[in] data user-defined data passed to `callbackFunction`.
   * @return Reference to this builder for method chaining.
   */
  RequestFlushBuilder& callbackData(RequestCallbackUserData data);

  /**
   * @brief Build and return the `RequestFlush`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestFlush>` object.
   */
  std::shared_ptr<RequestFlush> build() const;

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestFlush>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestFlush>` object.
   */
  operator std::shared_ptr<RequestFlush>() const;
};

/**
 * @brief Create a RequestFlushBuilder for constructing a `shared_ptr<ucxx::RequestFlush>`.
 *
 * @code{.cpp}
 *   auto req = ucxx::experimental::createRequestFlush(endpointOrWorker, flushData)
 *                .pythonFuture(true)
 *                .build();
 * @endcode
 *
 * @param[in] endpointOrWorker  the parent component (required).
 * @param[in] requestData       container of the flush request data (required).
 * @return A RequestFlushBuilder object that can be used to set optional parameters.
 */
inline RequestFlushBuilder createRequestFlush(std::shared_ptr<Component> endpointOrWorker,
                                              data::Flush requestData)
{
  return RequestFlushBuilder(std::move(endpointOrWorker), std::move(requestData));
}

}  // namespace experimental

}  // namespace ucxx
