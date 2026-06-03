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
class RequestTagMulti;

namespace experimental {

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestTagMulti>` objects.
 *
 * @copydoc ucxx_request_builder_pattern
 *
 * The `endpoint` and `requestData` are required and must be provided to
 * `createRequestTagMulti()`. The `pythonFuture()` method is optional.
 */
class RequestTagMultiBuilder : public RequestBuilderBase<RequestTagMultiBuilder> {
 private:
  std::shared_ptr<Endpoint> _endpoint;  ///< Parent endpoint (required)
  std::variant<data::TagMultiSend, data::TagMultiReceive>
    _requestData;  ///< Request-specific data (required)

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

  /**
   * @brief Implicit conversion operator to `shared_ptr<Request>`.
   *
   * @return The constructed request as `shared_ptr<ucxx::Request>`.
   */
  operator std::shared_ptr<Request>() const;
};

/**
 * @brief Create a RequestTagMultiBuilder for constructing a
 * `shared_ptr<ucxx::RequestTagMulti>`.
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
