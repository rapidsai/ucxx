/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

std::shared_ptr<::ucxx::Future> createPythonFuture(std::shared_ptr<::ucxx::Notifier> notifier);

std::shared_ptr<::ucxx::Notifier> createPythonNotifier();

std::shared_ptr<::ucxx::Worker> createPythonWorker(std::shared_ptr<ucxx::Context> context,
                                                   const bool enableDelayedSubmission,
                                                   const bool enablePythonFuture);

}  // namespace python

}  // namespace ucxx
