/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/config.h>
#include <ucxx/constructors.h>

namespace ucxx {

class Worker;

class Context : public Component {
 private:
  ucp_context_h _handle{nullptr};  ///< The UCP context handle
  Config _config{{}};              ///< UCP context configuration variables
  uint64_t _featureFlags{0};       ///< Feature flags used to construct UCP context
  bool _cudaSupport{false};        ///< Whether CUDA support is enabled

  /**
   * @brief Private constructor of `shared_ptr<ucxx::Context>`.
   *
   * This is the internal implementation of `ucxx::Context` constructor, made private not
   * to be called directly. Instead the user should call `ucxx::createContext()`.
   *
   * @param[in] ucxConfig configurations overriding `UCX_*` defaults and environment
   *                      variables.
   * @param[in] featureFlags feature flags to be used at UCP context construction time.
   */
  Context(const ConfigMap ucxConfig, const uint64_t featureFlags);

 public:
  static constexpr uint64_t defaultFeatureFlags =
    UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP | UCP_FEATURE_STREAM | UCP_FEATURE_AM | UCP_FEATURE_RMA;

  Context()               = delete;
  Context(const Context&) = delete;
  Context& operator=(Context const&) = delete;
  Context(Context&& o)               = delete;
  Context& operator=(Context&& o) = delete;

  /**
   * @brief Constructor of `shared_ptr<ucxx::Context>`.
   *
   * The constructor for a `shared_ptr<ucxx::Context>` object. The default constructor is
   * made private to ensure all UCXX objects are shared pointers for correct
   * lifetime management.
   *
   * @code{.cpp}
   *   auto context = ucxx::createContext({}, UCP_FEATURE_WAKEUP | UCP_FEATURE_TAG);
   * @endcode
   *
   * @param[in] ucxConfig configurations overriding `UCX_*` defaults and environment
   *                      variables.
   * @param[in] featureFlags feature flags to be used at UCP context construction time.
   * @return The `shared_ptr<ucxx::Context>` object
   */
  friend std::shared_ptr<Context> createContext(ConfigMap ucxConfig, const uint64_t featureFlags);

  /**
   * @brief `ucxx::Context` destructor
   */
  ~Context();

  /**
   * @brief Get the context configuration.
   *
   * The context configuration is a `ConfigMap` containing entries of the UCX variables that were
   * set upon creation of the UCP context. Only those variables known to UCP can be acquired.
   *
   * @code{.cpp}
   *   // context is `std::shared_ptr<ucxx::Context>`
   *   auto contextConfig = context->getConfig();
   * @endcode
   *
   * @return A `ConfigMap` corresponding to the context's configuration.
   */
  ConfigMap getConfig();

  /**
   * @brief Get the underlying `ucp_context_h` handle
   *
   * Lifetime of the `ucp_context_h` handle is managed by the `ucxx::Context`
   * object and its ownership is non-transferrable. Once the `ucxx::Context`
   * is destroyed the handle is not valid anymore, it is the user's
   * responsibility to ensure the owner's lifetime while using the handle.
   *
   * @code{.cpp}
   *   // context is `std::shared_ptr<ucxx::Context>`
   *   ucp_context_h contextHandle = context->getHandle();
   * @endcode
   *
   * @return The underlying `ucp_context_h` handle
   */
  ucp_context_h getHandle();

  /**
   * @brief Get information from UCP context.
   *
   * Get information from UCP context, including memory domains, transport
   * resources, and other useful information. This method is a wrapper to
   * `ucp_context_print_info`.
   *
   * @code{.cpp}
   *   // context is `std::shared_ptr<ucxx::Context>`
   *   auto contextInfo = context->getInfo();
   * @endcode
   *
   * @return String containing context information
   */
  std::string getInfo();

  /**
   * @brief Get feature flags that were used to construct the UCP context.
   *
   * Get feature flags that were used to construct the UCP context, this has
   * the same value that was specified by the user when creating the
   * `ucxx::Context` object.
   *
   * @code{.cpp}
   *   // context is `std::shared_ptr<ucxx::Context>`
   *   uint64_t contextFeatureFlags= context->getFeatureFlags();
   * @endcode
   *
   * @return Feature flags for this context
   */
  uint64_t getFeatureFlags() const;

  /**
   * @brief Create a new `ucxx::Worker`.
   *
   * Create a new `ucxx::Worker` as a child of the current `ucxx::Context`.
   * The `ucxx::Context` will retain ownership of the `ucxx::Worker` and will
   * not be destroyed until all `ucxx::Worker` objects are destroyed first.
   *
   * @code{.cpp}
   *   // context is `std::shared_ptr<ucxx::Context>`
   *   auto worker = context->createWorker(true);
   * @endcode
   *
   * @param[in] enableDelayedSubmission whether the worker should delay
   *                                    transfer requests to the worker thread.
   * @return Shared pointer to the `ucxx::Worker` object.
   */
  std::shared_ptr<Worker> createWorker(const bool enableDelayedSubmission = false);
};

}  // namespace ucxx
