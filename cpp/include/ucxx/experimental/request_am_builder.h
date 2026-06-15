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
class Request;
class RequestAm;

namespace experimental {

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestAm>` objects.
 *
 * @copydoc ucxx_request_builder_pattern
 *
 * The `endpoint` and `requestData` are required and must be provided to
 * `createRequestAm()`.
 *
 * @code{.cpp}
 *   auto req = ucxx::experimental::createRequestAm(endpoint, amSendData)
 *                .pythonFuture(true)
 *                .callbackFunction(callback)
 *                .callbackData(callbackData)
 *                .build();
 *
 *   std::shared_ptr<ucxx::RequestAm> amReq =
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
  [[nodiscard]] std::shared_ptr<RequestAm> build();

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestAm>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestAm>` object.
   */
  operator std::shared_ptr<RequestAm>();

  /**
   * @brief Implicit conversion operator to `shared_ptr<Request>`.
   *
   * @return The constructed request as `shared_ptr<ucxx::Request>`.
   */
  operator std::shared_ptr<Request>();
};

/**
 * @brief Create a RequestAmBuilder for constructing a `shared_ptr<ucxx::RequestAm>`.
 *
 * @param[in] endpoint     the parent endpoint (required).
 * @param[in] requestData  container of the specified message type (required).
 * @return A RequestAmBuilder object that can be used to set optional parameters.
 */
[[nodiscard]] inline RequestAmBuilder createRequestAm(
  std::shared_ptr<Endpoint> endpoint, std::variant<data::AmSend, data::AmReceive> requestData)
{
  return RequestAmBuilder(std::move(endpoint), std::move(requestData));
}

}  // namespace experimental

}  // namespace ucxx
