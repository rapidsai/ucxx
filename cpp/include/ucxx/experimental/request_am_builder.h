/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <utility>
#include <variant>

#include <ucxx/experimental/request_builder_base.h>
#include <ucxx/request_data.h>

namespace ucxx {

// Forward declarations
class Endpoint;
class RequestAm;

namespace experimental {

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestAm>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::RequestAm>`, allowing
 * optional parameters to be specified via method chaining. Construction happens when the
 * builder expression completes (via implicit conversion) or when `build()` is called.
 *
 * The `endpoint` and `requestData` are required and must be provided to
 * `createRequestAm()`. The remaining methods are optional.
 *
 * @code{.cpp}
 *   // Minimal usage (only required args)
 *   auto req = ucxx::experimental::createRequestAm(endpoint, amSendData).build();
 *
 *   // With optional parameters
 *   auto req = ucxx::experimental::createRequestAm(endpoint, amSendData)
 *                .pythonFuture(true)
 *                .callbackFunction(myCallback)
 *                .callbackData(myData)
 *                .build();
 *
 *   // Using implicit conversion
 *   std::shared_ptr<ucxx::RequestAm> req =
 *     ucxx::experimental::createRequestAm(endpoint, amReceiveData);
 * @endcode
 */
class RequestAmBuilder : public RequestCallbackBuilderBase<RequestAmBuilder> {
 private:
  std::shared_ptr<Endpoint> _endpoint;                       ///< Parent endpoint (required)
  std::variant<data::AmSend, data::AmReceive> _requestData;  ///< Request-specific data (required)

 public:
  /**
   * @brief Constructor for `RequestAmBuilder` with required parameters.
   *
   * @param[in] endpoint     the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] requestData  container of the specified message type, including all
   *                         type-specific data.
   */
  explicit RequestAmBuilder(std::shared_ptr<Endpoint> endpoint,
                            std::variant<data::AmSend, data::AmReceive> requestData);

  /**
   * @brief Build and return the `RequestAm`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestAm>` object.
   */
  std::shared_ptr<RequestAm> build() const;

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestAm>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestAm>` object.
   */
  operator std::shared_ptr<RequestAm>() const;
};

/**
 * @brief Create a RequestAmBuilder for constructing a `shared_ptr<ucxx::RequestAm>`.
 *
 * @code{.cpp}
 *   auto req = ucxx::experimental::createRequestAm(endpoint, amSendData)
 *                .pythonFuture(true)
 *                .build();
 * @endcode
 *
 * @param[in] endpoint     the parent endpoint (required).
 * @param[in] requestData  container of the specified message type (required).
 * @return A RequestAmBuilder object that can be used to set optional parameters.
 */
inline RequestAmBuilder createRequestAm(std::shared_ptr<Endpoint> endpoint,
                                        std::variant<data::AmSend, data::AmReceive> requestData)
{
  return RequestAmBuilder(std::move(endpoint), std::move(requestData));
}

}  // namespace experimental

}  // namespace ucxx
