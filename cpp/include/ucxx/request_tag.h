/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>

namespace ucxx {

// Forward declarations
class RequestTag;
class RequestTagParams;

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

// Forward declare the factory function to be friended
template <typename... Options>
std::enable_if_t<detail::contains_type<request_tag_params::EndpointParam, Options...>::value &&
                   detail::contains_type<request_tag_params::RequestDataParam, Options...>::value &&
                   detail::has_unique_types<detail::remove_cvref<Options>...>::value,
                 std::shared_ptr<RequestTag>>
createRequestTag(Options&&... opts);

/**
 * @brief Parameter container for RequestTag creation
 *
 * This class aggregates all possible parameters for RequestTag creation and
 * provides default values for optional parameters.
 */
class RequestTagParams {
 public:
  std::shared_ptr<Component> endpointOrWorker;
  std::variant<data::TagSend, data::TagReceive> requestData;
  bool enablePythonFuture{false};
  RequestCallbackUserFunction callbackFunction{nullptr};
  RequestCallbackUserData callbackData{nullptr};
  std::string operationName{"tagOp"};

  // Parameter setters
  void set(const request_tag_params::EndpointParam& p) { endpointOrWorker = p.value; }

  void set(const request_tag_params::RequestDataParam& p)
  {
    std::visit(
      [this](const auto& data) {
        using T = std::decay_t<decltype(data)>;
        requestData.template emplace<T>(data);
      },
      p.value);
  }

  void set(const request_tag_params::EnablePythonFutureParam& p) { enablePythonFuture = p.value; }
  void set(const request_tag_params::CallbackFunctionParam& p) { callbackFunction = p.value; }
  void set(const request_tag_params::CallbackDataParam& p) { callbackData = p.value; }
  void set(const request_tag_params::OperationNameParam& p) { operationName = p.value; }
};

/**
 * @brief Send or receive a message with the UCX Tag API.
 *
 * Send or receive a message with the UCX Tag API, using non-blocking UCP calls
 * `ucp_tag_send_nbx` or `ucp_tag_recv_nbx`.
 */
class RequestTag : public Request {
 private:
  /**
   * @brief Private constructor of `ucxx::RequestTag`.
   *
   * This is the internal implementation of `ucxx::RequestTag` constructor, made private not
   * to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Endpoint::tagRecv()`
   * - `ucxx::Endpoint::tagSend()`
   * - `ucxx::Worker::tagRecv()`
   * - `ucxx::createRequestTag()`
   *
   * @throws ucxx::Error  if send is `true` and `endpointOrWorker` is not a
   *                      `std::shared_ptr<ucxx::Endpoint>`.
   *
   * @param[in] endpointOrWorker    the parent component, which may either be a
   *                                `std::shared_ptr<Endpoint>` or
   *                                `std::shared_ptr<Worker>`.
   * @param[in] requestData         container of the specified message type, including all
   *                                type-specific data.
   * @param[in] operationName       a human-readable operation name to help identifying
   *                                requests by their types when UCXX logging is enabled.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   */
  RequestTag(std::shared_ptr<Component> endpointOrWorker,
             const std::variant<data::TagSend, data::TagReceive>& requestData,
             const std::string& operationName,
             const bool enablePythonFuture                = false,
             RequestCallbackUserFunction callbackFunction = nullptr,
             RequestCallbackUserData callbackData         = nullptr);

  // Friend declarations for both createRequestTag functions
  friend std::shared_ptr<RequestTag> createRequestTag(
    std::shared_ptr<Component> endpointOrWorker,
    const std::variant<data::TagSend, data::TagReceive> requestData,
    const bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);

  // Friend the templated version
  template <typename... Options>
  friend std::enable_if_t<
    detail::contains_type<request_tag_params::EndpointParam, Options...>::value &&
      detail::contains_type<request_tag_params::RequestDataParam, Options...>::value &&
      detail::has_unique_types<detail::remove_cvref<Options>...>::value,
    std::shared_ptr<RequestTag>>
  createRequestTag(Options&&... opts);

 public:
  virtual void populateDelayedSubmission();

  /**
   * @brief Create and submit a tag request.
   *
   * This is the method that should be called to actually submit a tag request. It is meant
   * to be called from `populateDelayedSubmission()`, which is decided at the discretion of
   * `std::shared_ptr<ucxx::Worker>`. See `populateDelayedSubmission()` for more details.
   */
  void request();

  /**
   * @brief Callback executed by UCX when a tag send request is completed.
   *
   * Callback executed by UCX when a tag send request is completed, that will dispatch
   * `ucxx::Request::callback()`.
   *
   * WARNING: This is not intended to be called by the user, but it currently needs to be
   * a public method so that UCX may access it. In future changes this will be moved to
   * an internal object and remove this method from the public API.
   *
   * @param[in] request the UCX request pointer.
   * @param[in] status  the completion status of the request.
   * @param[in] arg     the pointer to the `ucxx::Request` object that created the
   *                    transfer, effectively `this` pointer as seen by `request()`.
   */
  static void tagSendCallback(void* request, ucs_status_t status, void* arg);

  /**
   * @brief Callback executed by UCX when a tag receive request is completed.
   *
   * Callback executed by UCX when a tag receive request is completed, that will dispatch
   * `ucxx::RequestTag::callback()`.
   *
   * WARNING: This is not intended to be called by the user, but it currently needs to be
   * a public method so that UCX may access it. In future changes this will be moved to
   * an internal object and remove this method from the public API.
   *
   * @param[in] request the UCX request pointer.
   * @param[in] status  the completion status of the request.
   * @param[in] info    information of the completed transfer provided by UCX, includes
   *                    length of message received used to verify for truncation.
   * @param[in] arg     the pointer to the `ucxx::Request` object that created the
   *                    transfer, effectively `this` pointer as seen by `request()`.
   */
  static void tagRecvCallback(void* request,
                              ucs_status_t status,
                              const ucp_tag_recv_info_t* info,
                              void* arg);

  /**
   * @brief Implementation of the tag receive request callback.
   *
   * Implementation of the tag receive request callback. Verify whether the message was
   * truncated and set that state if necessary, and finally dispatch
   * `ucxx::Request::callback()`.
   *
   * WARNING: This is not intended to be called by the user, but it currently needs to be
   * a public method so that UCX may access it. In future changes this will be moved to
   * an internal object and remove this method from the public API.
   *
   * @param[in] request the UCX request pointer.
   * @param[in] status  the completion status of the request.
   * @param[in] info    information of the completed transfer provided by UCX, includes
   *                    length of message received used to verify for truncation.
   */
  void callback(void* request, ucs_status_t status, const ucp_tag_recv_info_t* info);
};

// Implementation of the templated createRequestTag function
template <typename... Options>
std::enable_if_t<detail::contains_type<request_tag_params::EndpointParam, Options...>::value &&
                   detail::contains_type<request_tag_params::RequestDataParam, Options...>::value &&
                   detail::has_unique_types<detail::remove_cvref<Options>...>::value,
                 std::shared_ptr<RequestTag>>
createRequestTag(Options&&... opts)
{
  // Default values for optional parameters
  std::shared_ptr<Component> endpointOrWorker;
  std::optional<std::variant<data::TagSend, data::TagReceive>> requestData;
  bool enablePythonFuture                      = false;
  RequestCallbackUserFunction callbackFunction = nullptr;
  RequestCallbackUserData callbackData         = nullptr;
  std::string operationName                    = "tagOp";

  // Helper to set parameters
  auto setParam = [&](auto&& param) {
    using ParamType = std::decay_t<decltype(param)>;
    if constexpr (std::is_same_v<ParamType, request_tag_params::EndpointParam>) {
      endpointOrWorker = std::move(param.value);
    } else if constexpr (std::is_same_v<ParamType, request_tag_params::RequestDataParam>) {
      requestData.emplace(std::move(param.value));
    } else if constexpr (std::is_same_v<ParamType, request_tag_params::EnablePythonFutureParam>) {
      enablePythonFuture = param.value;
    } else if constexpr (std::is_same_v<ParamType, request_tag_params::CallbackFunctionParam>) {
      callbackFunction = param.value;
    } else if constexpr (std::is_same_v<ParamType, request_tag_params::CallbackDataParam>) {
      callbackData = param.value;
    } else if constexpr (std::is_same_v<ParamType, request_tag_params::OperationNameParam>) {
      operationName = std::move(param.value);
    }
  };

  // Set all parameters
  (setParam(std::forward<Options>(opts)), ...);

  // Ensure required parameters are present
  if (!endpointOrWorker || !requestData) {
    throw std::runtime_error("Missing required parameters for RequestTag creation");
  }

  // Create the RequestTag with the collected parameters
  auto req = std::shared_ptr<RequestTag>(new RequestTag(std::move(endpointOrWorker),
                                                        std::move(*requestData),
                                                        std::move(operationName),
                                                        enablePythonFuture,
                                                        callbackFunction,
                                                        callbackData));

  // Register delayed submission
  req->_worker->registerDelayedSubmission(
    req, std::bind(std::mem_fn(&Request::populateDelayedSubmission), req.get()));

  return req;
}

}  // namespace ucxx
