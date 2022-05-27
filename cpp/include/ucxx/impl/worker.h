/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <functional>
#include <mutex>
#include <queue>

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <ucxx/utils.h>
#include <ucxx/worker.h>

namespace ucxx {

UCXXWorker::UCXXWorker(std::shared_ptr<UCXXContext> context, const bool enableDelayedNotification)
{
  ucp_worker_params_t worker_params;

  if (context == nullptr || context->get_handle() == nullptr)
    throw std::runtime_error("UCXXContext not initialized");

  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
  assert_ucs_status(ucp_worker_create(context->get_handle(), &worker_params, &_handle));

  if (enableDelayedNotification) {
    ucxx_info("Worker %p created with delayed request notification", this);
    _delayedNotificationRequestCollection =
      std::make_shared<DelayedNotificationRequestCollection>();
  } else {
    ucxx_info("Worker %p created with immediate request notification", this);
  }

  setParent(std::dynamic_pointer_cast<UCXXComponent>(context));
}

void UCXXWorker::drainWorkerTagRecv()
{
  // TODO: Uncomment, requires specialized UCXXTransferTag
  // auto context = std::dynamic_pointer_cast<UCXXContext>(_parent);
  // if (!(context->get_feature_flags() & UCP_FEATURE_TAG)) return;

  // ucp_tag_message_h message;
  // ucp_tag_recv_info_t info;

  // while ((message = ucp_tag_probe_nb(_handle, 0, 0, 1, &info)) != NULL) {
  //   ucxx_debug("Draining tag receive messages, worker: %p, tag: 0x%lx, length: %lu",
  //              _handle,
  //              info.sender_tag,
  //              info.length);

  //   std::shared_ptr<ucxx_request_t> request = std::make_shared<ucxx_request_t>();
  //   ucp_request_param_t param               = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
  //                                                UCP_OP_ATTR_FIELD_DATATYPE |
  //                                                UCP_OP_ATTR_FIELD_USER_DATA,
  //                                .datatype  = ucp_dt_make_contig(1),
  //                                .user_data = request.get()};

  //   std::unique_ptr<char> buf = std::make_unique<char>(info.length);
  //   ucs_status_ptr_t status =
  //     ucp_tag_msg_recv_nbx(_handle, buf.get(), info.length, message, &param);
  //   request_wait(_handle, status, request.get(), "drain_tag_recv");

  //   while (request->status == UCS_INPROGRESS)
  //     progress();
  // }
}

std::shared_ptr<UCXXWorker> createWorker(std::shared_ptr<UCXXContext> context,
                                         const bool enableDelayedNotification)
{
  return std::shared_ptr<UCXXWorker>(new UCXXWorker(context, enableDelayedNotification));
}

UCXXWorker::~UCXXWorker()
{
  stopProgressThread();
#if UCXX_ENABLE_PYTHON
  _notifier->stopRequestNotifierThread();
#endif

  drainWorkerTagRecv();

  ucp_worker_destroy(_handle);

  if (_epollFileDescriptor >= 0) close(_epollFileDescriptor);
  if (_wakeFileDescriptor >= 0) close(_wakeFileDescriptor);
}

ucp_worker_h UCXXWorker::get_handle() { return _handle; }

void UCXXWorker::init_blocking_progress_mode()
{
  // In blocking progress mode, we create an epoll file
  // descriptor that we can wait on later.
  // We also introduce an additional eventfd to allow
  // canceling the wait.
  int err;

  // Return if blocking progress mode was already initialized
  if (_epollFileDescriptor >= 0) return;

  assert_ucs_status(ucp_worker_get_efd(_handle, &_workerFileDescriptor));

  arm();

  _epollFileDescriptor = epoll_create(1);
  if (_epollFileDescriptor == -1) throw std::ios_base::failure("epoll_create(1) returned -1");

  _wakeFileDescriptor = eventfd(0, EFD_NONBLOCK);
  if (_wakeFileDescriptor < 0) throw std::ios_base::failure("eventfd(0, EFD_NONBLOCK) returned -1");

  epoll_event workerEvent = {.events = EPOLLIN,
                             .data   = {
                               .fd = _workerFileDescriptor,
                             }};
  epoll_event wakeEvent   = {.events = EPOLLIN,
                           .data   = {
                             .fd = _wakeFileDescriptor,
                           }};

  err = epoll_ctl(_epollFileDescriptor, EPOLL_CTL_ADD, _workerFileDescriptor, &workerEvent);
  if (err != 0) throw std::ios_base::failure(std::string("epoll_ctl() returned " + err));
  err = epoll_ctl(_epollFileDescriptor, EPOLL_CTL_ADD, _wakeFileDescriptor, &wakeEvent);
  if (err != 0) throw std::ios_base::failure(std::string("epoll_ctl() returned " + err));
}

bool UCXXWorker::arm()
{
  ucs_status_t status = ucp_worker_arm(_handle);
  if (status == UCS_ERR_BUSY) return false;
  assert_ucs_status(status);
  return true;
}

bool UCXXWorker::progress_worker_event()
{
  int ret;
  epoll_event ev;

  if (progress_once()) return true;

  if ((_epollFileDescriptor == -1) || !arm()) return false;

  do {
    ret = epoll_wait(_epollFileDescriptor, &ev, 1, -1);
  } while ((ret == -1) && (errno == EINTR || errno == EAGAIN));

  return false;
}

void UCXXWorker::wakeProgressEvent()
{
  if (_wakeFileDescriptor < 0)
    throw std::ios_base::failure(std::string(
      "attempt to wake progress event, but blocking progress mode was not initialized"));
  int err = eventfd_write(_wakeFileDescriptor, 1);
  if (err < 0) throw std::ios_base::failure(std::string("eventfd_write() returned " + err));
}

bool UCXXWorker::wait_progress()
{
  assert_ucs_status(ucp_worker_wait(_handle));
  return progress_once();
}

bool UCXXWorker::progress_once() { return ucp_worker_progress(_handle) != 0; }

void UCXXWorker::progress()
{
  while (progress_once())
    ;
}

void UCXXWorker::registerNotificationRequest(NotificationRequestCallbackType callback)
{
  if (_delayedNotificationRequestCollection == nullptr) {
    callback();
  } else {
    _delayedNotificationRequestCollection->registerRequest(callback);

    /* Waking the progress event is needed here because the UCX request is
     * not dispatched immediately. Thus we must wake the progress task so
     * it will ensure the request is dispatched.
     */
    wakeProgressEvent();
  }
}

void UCXXWorker::populatePythonFuturesPool()
{
#if UCXX_ENABLE_PYTHON
  ucxx_trace_req("populatePythonFuturesPool: %p %p", this, shared_from_this().get());
  // If the pool goes under half expected size, fill it up again.
  if (_pythonFuturesPool.size() < 50) {
    std::lock_guard<std::mutex> lock(_pythonFuturesPoolMutex);
    while (_pythonFuturesPool.size() < 100)
      _pythonFuturesPool.emplace(std::make_shared<PythonFuture>(_notifier));
  }
#else
  std::runtime_error("Python support not enabled, please compiled with -DUCXX_ENABLE_PYTHON 1");
#endif
}

std::shared_ptr<PythonFuture> UCXXWorker::getPythonFuture()
{
#if UCXX_ENABLE_PYTHON
  if (_pythonFuturesPool.size() == 0) {
    ucxx_warn(
      "No Python Futures available during getPythonFuture(), make sure the "
      "Notifier Thread is running and calling populatePythonFuturesPool() "
      "periodically. Filling futures pool now, but this is inefficient.");
    populatePythonFuturesPool();
  }

  std::shared_ptr<PythonFuture> ret{nullptr};
  {
    std::lock_guard<std::mutex> lock(_pythonFuturesPoolMutex);
    ret = _pythonFuturesPool.front();
    _pythonFuturesPool.pop();
  }
  ucxx_trace_req("getPythonFuture: %p %p", ret.get(), ret->getHandle());
  return ret;
#else
  std::runtime_error("Python support not enabled, please compiled with -DUCXX_ENABLE_PYTHON 1");
  return nullptr;
#endif
}

bool UCXXWorker::waitRequestNotifier()
{
#if UCXX_ENABLE_PYTHON
  return _notifier->waitRequestNotifier();
#else
  std::runtime_error("Python support not enabled, please compiled with -DUCXX_ENABLE_PYTHON 1");
  return false;
#endif
}

void UCXXWorker::runRequestNotifier()
{
#if UCXX_ENABLE_PYTHON
  _notifier->runRequestNotifier();
#else
  std::runtime_error("Python support not enabled, please compiled with -DUCXX_ENABLE_PYTHON 1");
#endif
}

void UCXXWorker::stopRequestNotifierThread()
{
#if UCXX_ENABLE_PYTHON
  _notifier->stopRequestNotifierThread();
#else
  std::runtime_error("Python support not enabled, please compiled with -DUCXX_ENABLE_PYTHON 1");
#endif
}

void UCXXWorker::setProgressThreadStartCallback(std::function<void(void*)> callback,
                                                void* callbackArg)
{
  _progressThreadStartCallback    = callback;
  _progressThreadStartCallbackArg = callbackArg;
}

void UCXXWorker::startProgressThread(const bool pollingMode)
{
  if (_progressThread) {
    ucxx_warn("Worker progress thread already running");
    return;
  }

  if (pollingMode) init_blocking_progress_mode();
  auto progressFunction = pollingMode ? std::bind(&UCXXWorker::progress_worker_event, this)
                                      : std::bind(&UCXXWorker::progress_once, this);

  _progressThread =
    std::make_shared<UCXXWorkerProgressThread>(pollingMode,
                                               progressFunction,
                                               _progressThreadStartCallback,
                                               _progressThreadStartCallbackArg,
                                               _delayedNotificationRequestCollection);
}

void UCXXWorker::stopProgressThread()
{
  if (!_progressThread) {
    ucxx_warn("Worker progress thread not running or already stopped");
    return;
  }

  if (_progressThread->pollingMode()) wakeProgressEvent();
  _progressThread = nullptr;
}

inline size_t UCXXWorker::cancelInflightRequests()
{
  // Fast path when no requests have been scheduled for cancelation
  if (_inflightRequestsToCancel->size() == 0) return 0;

  size_t total = 0;
  std::lock_guard<std::mutex> lock(_inflightMutex);

  for (auto& r : *_inflightRequestsToCancel) {
    if (auto request = r.second.lock()) {
      request->cancel();
      ++total;
    }
  }

  _inflightRequestsToCancel->clear();
  return total;
}

void UCXXWorker::scheduleRequestCancel(inflight_requests_t inflightRequests)
{
  std::lock_guard<std::mutex> lock(_inflightMutex);
  _inflightRequestsToCancel->insert(inflightRequests->begin(), inflightRequests->end());
}

bool UCXXWorker::tagProbe(ucp_tag_t tag)
{
  ucp_tag_recv_info_t info;
  ucp_tag_message_h tag_message = ucp_tag_probe_nb(_handle, tag, -1, 0, &info);

  return tag_message != NULL;
}

std::shared_ptr<UCXXAddress> UCXXWorker::getAddress()
{
  auto worker  = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
  auto address = ucxx::createAddressFromWorker(worker);
  return address;
}

std::shared_ptr<UCXXEndpoint> UCXXWorker::createEndpointFromHostname(std::string ip_address,
                                                                     uint16_t port,
                                                                     bool endpoint_error_handling)
{
  auto worker = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
  auto endpoint =
    ucxx::createEndpointFromHostname(worker, ip_address, port, endpoint_error_handling);
  return endpoint;
}

std::shared_ptr<UCXXEndpoint> UCXXWorker::createEndpointFromWorkerAddress(
  std::shared_ptr<UCXXAddress> address, bool endpoint_error_handling)
{
  auto worker   = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
  auto endpoint = ucxx::createEndpointFromWorkerAddress(worker, address, endpoint_error_handling);
  return endpoint;
}

std::shared_ptr<UCXXListener> UCXXWorker::createListener(uint16_t port,
                                                         ucp_listener_conn_callback_t callback,
                                                         void* callback_args)
{
  auto worker   = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
  auto listener = ucxx::createListener(worker, port, callback, callback_args);
  return listener;
}

}  // namespace ucxx
