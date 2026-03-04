/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <utility>

namespace ucxx {

// Forward declarations
class Context;
class Worker;

namespace experimental {

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::Worker>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::Worker>`, allowing
 * optional parameters to be specified via method chaining. Construction happens when the
 * builder expression completes (via implicit conversion) or when `build()` is called.
 *
 * The `context` is required and must be provided to `createWorker()`.
 * The `enableDelayedSubmission()` and `enableFuture()` methods are optional.
 *
 * @code{.cpp}
 *   // Minimal usage (no options enabled)
 *   auto worker = ucxx::experimental::createWorker(context).build();
 *
 *   // With optional parameters
 *   auto worker = ucxx::experimental::createWorker(context)
 *                   .enableDelayedSubmission(true)
 *                   .enableFuture(true)
 *                   .build();
 *
 *   // Using implicit conversion
 *   std::shared_ptr<ucxx::Worker> worker = ucxx::experimental::createWorker(context);
 * @endcode
 */
class WorkerBuilder {
 private:
  std::shared_ptr<Context> _context;     ///< UCXX context (required)
  bool _enableDelayedSubmission{false};  ///< Enable delayed submission to progress thread
  bool _enableFuture{false};             ///< Enable future support

 public:
  /**
   * @brief Constructor for `WorkerBuilder` with required context.
   *
   * @param[in] context context from which the worker will be created (required).
   */
  explicit WorkerBuilder(std::shared_ptr<Context> context);

  /**
   * @brief Configure delayed submission to the progress thread.
   *
   * @param[in] enable whether delayed submission is enabled (default: true).
   * @return Reference to this builder for method chaining.
   */
  WorkerBuilder& delayedSubmission(bool enable = true);

  /**
   * @brief Configure Python future support.
   *
   * @param[in] enable whether Python futures are enabled (default: true).
   * @return Reference to this builder for method chaining.
   */
  WorkerBuilder& pythonFuture(bool enable = true);

  /**
   * @brief Build and return the `Worker`.
   *
   * This method constructs the `Worker` with the specified parameters and returns it.
   * Each call to build() creates a new `Worker` instance with the current parameters.
   *
   * @return The constructed `shared_ptr<ucxx::Worker>` object.
   */
  std::shared_ptr<Worker> build() const;

  /**
   * @brief Implicit conversion operator to `shared_ptr<Worker>`.
   *
   * This operator enables automatic construction of the `Worker` when the builder
   * is used in a context requiring a `shared_ptr<Worker>`. This allows seamless
   * use with `auto` variables.
   *
   * @return The constructed `shared_ptr<ucxx::Worker>` object.
   */
  operator std::shared_ptr<Worker>() const;
};

/**
 * @brief Create a WorkerBuilder for constructing a `shared_ptr<ucxx::Worker>`.
 *
 * This function returns a builder object that allows setting optional context parameters
 * via method chaining. The context is required and must be provided as an argument. The
 * actual worker is constructed when the builder is converted to a `shared_ptr<Worker>`
 * or when `build()` is called.
 *
 * @code{.cpp}
 *   auto worker = ucxx::experimental::createWorker(context)
 *                   .enableDelayedSubmission()
 *                   .build();
 * // Minimal usage (only context required)
 * auto worker = ucxx::experimental::createWorker().build();
 *
 * // With optional enableDelayedSubmission
 * auto worker = ucxx::experimental::createContext(UCP_FEATURE_TAG)
 *                 .enableDelayedSubmission()
 *                 .build();
 *
 * // Using implicit conversion
 * std::shared_ptr<ucxx::Worker> worker = ucxx::experimental::createWorker();
 *
 * @endcode
 *
 * @param[in] context context from which the worker will be created (required).
 * @return A WorkerBuilder object that can be used to set optional parameters.
 */
inline WorkerBuilder createWorker(std::shared_ptr<Context> context)
{
  return WorkerBuilder(std::move(context));
}

}  // namespace experimental

}  // namespace ucxx
