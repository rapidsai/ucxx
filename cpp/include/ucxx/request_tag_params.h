/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <ucxx/component.h>
#include <ucxx/request.h>
#include <ucxx/request_data.h>
#include <ucxx/typedefs.h>

namespace ucxx {

namespace detail {
// Helper to remove const, volatile and reference qualifiers (C++17 compatible version of
// remove_cvref)
template <typename T>
using remove_cvref = std::remove_reference_t<std::remove_cv_t<T>>;

// Type traits to detect parameter types
template <typename T>
struct is_endpoint_param : std::false_type {};

template <typename T>
struct is_request_data_param : std::false_type {};

// Helper to check if parameter pack contains a type
template <typename T, typename... Args>
struct contains_type : std::disjunction<std::is_same<T, remove_cvref<Args>>...> {};

// Helper to ensure no duplicate parameter types
template <typename... Args>
struct has_unique_types;

template <>
struct has_unique_types<> : std::true_type {};

template <typename T, typename... Rest>
struct has_unique_types<T, Rest...> {
  static constexpr bool value =
    (!std::disjunction<std::is_same<T, Rest>...>::value) && has_unique_types<Rest...>::value;
};
}  // namespace detail

/**
 * @brief Parameter tag types for RequestTag creation
 *
 * These types provide a type-safe way to pass named parameters to createRequestTag.
 * Each type wraps a specific parameter and provides a clear name at the call site.
 */
namespace request_tag_params {

/**
 * @brief Parameter wrapper for endpoint or worker component
 */
struct EndpointParam {
  std::shared_ptr<Component> value;
  explicit EndpointParam(std::shared_ptr<Component> ep) : value(std::move(ep)) {}
};

/**
 * @brief Parameter wrapper for request data (TagSend or TagReceive)
 */
struct RequestDataParam {
  std::variant<data::TagSend, data::TagReceive> value;

  explicit RequestDataParam(const data::TagSend& send) : value(send) {}

  explicit RequestDataParam(const data::TagReceive& recv) : value(recv) {}

  explicit RequestDataParam(const std::variant<data::TagSend, data::TagReceive>& data) : value(data)
  {
  }
};

/**
 * @brief Parameter wrapper for Python future enablement
 */
struct EnablePythonFutureParam {
  bool value;
  explicit EnablePythonFutureParam(bool enable) : value(enable) {}
};

/**
 * @brief Parameter wrapper for callback function
 */
struct CallbackFunctionParam {
  RequestCallbackUserFunction value;
  explicit CallbackFunctionParam(RequestCallbackUserFunction fn) : value(fn) {}
};

/**
 * @brief Parameter wrapper for callback data
 */
struct CallbackDataParam {
  RequestCallbackUserData value;
  explicit CallbackDataParam(RequestCallbackUserData data) : value(data) {}
};

/**
 * @brief Parameter wrapper for operation name
 */
struct OperationNameParam {
  std::string value;
  explicit OperationNameParam(std::string name) : value(std::move(name)) {}
};

}  // namespace request_tag_params

// Complete the type trait specializations after parameter types are defined
namespace detail {
template <>
struct is_endpoint_param<request_tag_params::EndpointParam> : std::true_type {};

template <>
struct is_request_data_param<request_tag_params::RequestDataParam> : std::true_type {};
}  // namespace detail

// Forward declarations
class RequestTag;

// Forward declare the factory function to be friended
template <typename... Options>
std::enable_if_t<detail::contains_type<request_tag_params::EndpointParam, Options...>::value &&
                   detail::contains_type<request_tag_params::RequestDataParam, Options...>::value &&
                   detail::has_unique_types<detail::remove_cvref<Options>...>::value,
                 std::shared_ptr<RequestTag>>
createRequestTag(Options&&... opts);

}  // namespace ucxx