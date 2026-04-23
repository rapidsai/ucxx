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
class Component;
class RequestTag;

namespace experimental {

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestTag>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::RequestTag>`, allowing
 * optional parameters to be specified via method chaining. Construction happens when the
 * builder expression completes (via implicit conversion) or when `build()` is called.
 *
 * The `endpointOrWorker` and `requestData` are required and must be provided to
 * `createRequestTag()`. The remaining methods are optional.
 *
 * @code{.cpp}
 *   // Minimal usage (only required args)
 *   auto req = ucxx::experimental::createRequestTag(endpointOrWorker, tagSendData).build();
 *
 *   // With optional parameters
 *   auto req = ucxx::experimental::createRequestTag(endpointOrWorker, tagSendData)
 *                .pythonFuture(true)
 *                .callbackFunction(myCallback)
 *                .callbackData(myData)
 *                .build();
 *
 *   // Using implicit conversion
 *   std::shared_ptr<ucxx::RequestTag> req =
 *     ucxx::experimental::createRequestTag(endpointOrWorker, tagRecvData);
 * @endcode
 */
class RequestTagBuilder {
 private:
  std::shared_ptr<Component> _endpointOrWorker;           ///< Parent endpoint or worker (required)
  std::variant<data::TagSend, data::TagReceive, data::TagReceiveWithHandle>
    _requestData;                                         ///< Request-specific data (required)
  bool _enablePythonFuture{false};                        ///< Enable Python future support
  RequestCallbackUserFunction _callbackFunction{nullptr}; ///< User callback on completion
  RequestCallbackUserData _callbackData{nullptr};         ///< Data passed to callback

 public:
  /**
   * @brief Constructor for `RequestTagBuilder` with required parameters.
   *
   * @param[in] endpointOrWorker  the parent component, which may be a
   *                              `std::shared_ptr<Endpoint>` or `std::shared_ptr<Worker>`.
   * @param[in] requestData       container of the specified message type, including all
   *                              type-specific data.
   */
  explicit RequestTagBuilder(
    std::shared_ptr<Component> endpointOrWorker,
    std::variant<data::TagSend, data::TagReceive, data::TagReceiveWithHandle> requestData);

  /**
   * @brief Configure Python future support.
   *
   * @param[in] enable whether a Python future should be created and notified (default: true).
   * @return Reference to this builder for method chaining.
   */
  RequestTagBuilder& pythonFuture(bool enable = true);

  /**
   * @brief Set the user-defined callback function to call upon completion.
   *
   * @param[in] fn user-defined callback function.
   * @return Reference to this builder for method chaining.
   */
  RequestTagBuilder& callbackFunction(RequestCallbackUserFunction fn);

  /**
   * @brief Set the user-defined data to pass to the callback function.
   *
   * @param[in] data user-defined data passed to `callbackFunction`.
   * @return Reference to this builder for method chaining.
   */
  RequestTagBuilder& callbackData(RequestCallbackUserData data);

  /**
   * @brief Build and return the `RequestTag`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestTag>` object.
   */
  std::shared_ptr<RequestTag> build() const;

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestTag>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestTag>` object.
   */
  operator std::shared_ptr<RequestTag>() const;
};

/**
 * @brief Create a RequestTagBuilder for constructing a `shared_ptr<ucxx::RequestTag>`.
 *
 * @code{.cpp}
 *   auto req = ucxx::experimental::createRequestTag(endpointOrWorker, tagSendData)
 *                .pythonFuture(true)
 *                .build();
 * @endcode
 *
 * @param[in] endpointOrWorker  the parent component (required).
 * @param[in] requestData       container of the specified message type (required).
 * @return A RequestTagBuilder object that can be used to set optional parameters.
 */
inline RequestTagBuilder createRequestTag(
  std::shared_ptr<Component> endpointOrWorker,
  std::variant<data::TagSend, data::TagReceive, data::TagReceiveWithHandle> requestData)
{
  return RequestTagBuilder(std::move(endpointOrWorker), std::move(requestData));
}

}  // namespace experimental

}  // namespace ucxx
