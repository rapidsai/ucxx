/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <utility>

#include <ucxx/experimental/request_builder_base.h>
#include <ucxx/request_data.h>

namespace ucxx {

// Forward declarations
class Component;
class Request;
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
class RequestFlushBuilder : public RequestCallbackBuilderBase<RequestFlushBuilder> {
 private:
  std::shared_ptr<Component> _endpointOrWorker;  ///< Parent endpoint or worker (required)
  data::Flush _requestData;                      ///< Request-specific data (required)

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

  /**
   * @brief Implicit conversion operator to `shared_ptr<Request>`.
   *
   * @return The constructed request as `shared_ptr<ucxx::Request>`.
   */
  operator std::shared_ptr<Request>() const;
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
