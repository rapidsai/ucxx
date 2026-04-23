/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <ucxx/experimental/builder_utils.h>
#include <ucxx/experimental/worker_builder.h>
#include <ucxx/worker.h>

namespace ucxx {

namespace experimental {

struct WorkerBuilder::Impl {
  std::shared_ptr<Context> context;
  bool enableDelayedSubmission{false};
  bool enableFuture{false};

  explicit Impl(std::shared_ptr<Context> ctx) : context(std::move(ctx)) {}
};

WorkerBuilder::WorkerBuilder(std::shared_ptr<Context> context)
  : _impl(std::make_unique<Impl>(std::move(context)))
{
}

UCXX_BUILDER_PIMPL_DEFAULTS(WorkerBuilder, Worker)

WorkerBuilder& WorkerBuilder::delayedSubmission(bool enable)
{
  _impl->enableDelayedSubmission = enable;
  return *this;
}

WorkerBuilder& WorkerBuilder::pythonFuture(bool enable)
{
  _impl->enableFuture = enable;
  return *this;
}

std::shared_ptr<Worker> WorkerBuilder::build() const
{
  return ucxx::createWorker(_impl->context, _impl->enableDelayedSubmission, _impl->enableFuture);
}

}  // namespace experimental

}  // namespace ucxx
