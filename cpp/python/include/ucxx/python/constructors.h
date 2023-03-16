/**
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>

#include <ucxx/buffer.h>

namespace ucxx {

class Context;
class Future;
class Notifier;
class Worker;

namespace python {

std::shared_ptr<::ucxx::Future> createFuture(std::shared_ptr<::ucxx::Notifier> notifier);

std::shared_ptr<::ucxx::Notifier> createNotifier();

std::shared_ptr<::ucxx::Worker> createWorker(std::shared_ptr<ucxx::Context> context,
                                             const bool enableDelayedSubmission,
                                             const bool enablePythonFuture);

}  // namespace python

}  // namespace ucxx
