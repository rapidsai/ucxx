/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <utility>
#include <variant>

#include <ucxx/request_builder_base.h>
#include <ucxx/request_data.h>

namespace ucxx {

// Forward declarations
class Endpoint;
class Request;
class RequestMem;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestMem>` objects.
 *
 * @copydoc ucxx_request_builder_pattern
 *
 * The `endpoint` and `requestData` are required and must be provided to
 * `requestMemBuilder()`.
 *
 * @code{.cpp}
 *   auto req = ucxx::requestMemBuilder(endpoint, memPutData)
 *                .pythonFuture(true)
 *                .callbackFunction(callback)
 *                .callbackData(callbackData)
 *                .build();
 *
 *   std::shared_ptr<ucxx::RequestMem> memReq =
 *     ucxx::requestMemBuilder(endpoint, memGetData);
 * @endcode
 */
class RequestMemBuilder : public RequestCallbackBuilderBase<RequestMemBuilder> {
 private:
  std::shared_ptr<Endpoint> _endpoint;                    ///< Parent endpoint (required)
  std::variant<data::MemPut, data::MemGet> _requestData;  ///< Request-specific data (required)

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
   * @brief Build and return the `RequestMem`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestMem>` object.
   */
  [[nodiscard]] std::shared_ptr<RequestMem> build();

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestMem>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestMem>` object.
   */
  operator std::shared_ptr<RequestMem>();

  /**
   * @brief Implicit conversion operator to `shared_ptr<Request>`.
   *
   * @return The constructed request as `shared_ptr<ucxx::Request>`.
   */
  operator std::shared_ptr<Request>();
};

/**
 * @brief Create a RequestMemBuilder for constructing a `shared_ptr<ucxx::RequestMem>`.
 *
 * @param[in] endpoint     the parent endpoint (required).
 * @param[in] requestData  container of the specified message type (required).
 * @return A RequestMemBuilder object that can be used to set optional parameters.
 */
[[nodiscard]] inline RequestMemBuilder requestMemBuilder(
  std::shared_ptr<Endpoint> endpoint, std::variant<data::MemPut, data::MemGet> requestData)
{
  return RequestMemBuilder(std::move(endpoint), std::move(requestData));
}

}  // namespace ucxx
