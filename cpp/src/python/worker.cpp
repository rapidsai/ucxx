/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <functional>
#include <ios>
#include <memory>
#include <mutex>

#include <Python.h>

#include <ucxx/python/future.h>
#include <ucxx/python/worker.h>
#include <ucxx/request_tag.h>
#include <ucxx/utils/python.h>

namespace ucxx {

namespace python {

Worker::Worker(std::shared_ptr<Context> context,
               const bool enableDelayedSubmission,
               const bool enableFuture)
  : ::ucxx::Worker(context, enableDelayedSubmission)
{
  if (_enableFuture) _notifier = createPythonNotifier();
}

std::shared_ptr<::ucxx::Worker> createPythonWorker(std::shared_ptr<Context> context,
                                                   const bool enableDelayedSubmission,
                                                   const bool enableFuture)
{
  return std::shared_ptr<::ucxx::Worker>(
    new ::ucxx::python::Worker(context, enableDelayedSubmission, enableFuture));
}

void Worker::populateFuturesPool()
{
  if (_enableFuture) {
    ucxx_trace_req("populateFuturesPool: %p %p", this, shared_from_this().get());
    // If the pool goes under half expected size, fill it up again.
    if (_futuresPool.size() < 50) {
      std::lock_guard<std::mutex> lock(_futuresPoolMutex);
      PyGILState_STATE state = PyGILState_Ensure();
      while (_futuresPool.size() < 100)
        _futuresPool.emplace(createPythonFuture(_notifier));
      PyGILState_Release(state);
    }
  } else {
    throw std::runtime_error(
      "Worker future support disabled, please set enableFuture=true when creating the "
      "Worker to use this method.");
  }
}

std::shared_ptr<::ucxx::Future> Worker::getFuture()
{
  if (_enableFuture) {
    if (_futuresPool.size() == 0) {
      ucxx_warn(
        "No Futures available during getFuture(), make sure the Notifier is running "
        "running and calling populateFuturesPool() periodically. Filling futures pool "
        "now, but this may be inefficient.");
      populateFuturesPool();
    }

    std::shared_ptr<::ucxx::Future> ret{nullptr};
    {
      std::lock_guard<std::mutex> lock(_futuresPoolMutex);
      ret = _futuresPool.front();
      _futuresPool.pop();
    }
    ucxx_trace_req("getFuture: %p %p", ret.get(), ret->getHandle());
    return std::dynamic_pointer_cast<::ucxx::Future>(ret);
  } else {
    throw std::runtime_error(
      "Worker future support disabled, please set enableFuture=true when creating the "
      "Worker to use this method.");
    return nullptr;
  }
}

RequestNotifierWaitState Worker::waitRequestNotifier(uint64_t periodNs)
{
  if (_enableFuture) {
    return _notifier->waitRequestNotifier(periodNs);
  } else {
    throw std::runtime_error(
      "Worker future support disabled, please set enableFuture=true when creating the "
      "Worker to use this method.");
  }
}

void Worker::runRequestNotifier()
{
  if (_enableFuture) {
    _notifier->runRequestNotifier();
  } else {
    throw std::runtime_error(
      "Worker future support disabled, please set enableFuture=true when creating the "
      "Worker to use this method.");
  }
}

void Worker::stopRequestNotifierThread()
{
  if (_enableFuture) {
    _notifier->stopRequestNotifierThread();
  } else {
    throw std::runtime_error(
      "Worker future support disabled, please set enableFuture=true when creating the "
      "Worker to use this method.");
  }
}

}  // namespace python

}  // namespace ucxx
