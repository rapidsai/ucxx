/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <ucxx/experimental/worker_builder.h>
#include <ucxx/worker.h>

namespace ucxx {

namespace experimental {

WorkerBuilder::WorkerBuilder(std::shared_ptr<Context> context) : _context(std::move(context)) {}

WorkerBuilder& WorkerBuilder::delayedSubmission(bool enable)
{
  _enableDelayedSubmission = enable;
  return *this;
}

WorkerBuilder& WorkerBuilder::pythonFuture(bool enable)
{
  _enableFuture = enable;
  return *this;
}

std::shared_ptr<Worker> WorkerBuilder::build() const
{
  return std::shared_ptr<Worker>(new Worker(_context, _enableDelayedSubmission, _enableFuture));
}

WorkerBuilder::operator std::shared_ptr<Worker>() const
{
  return std::shared_ptr<Worker>(new Worker(_context, _enableDelayedSubmission, _enableFuture));
}

}  // namespace experimental

}  // namespace ucxx
