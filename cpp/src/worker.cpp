/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <functional>
#include <ios>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <ucxx/buffer.h>
#include <ucxx/internal/request_am.h>
#include <ucxx/request_am.h>
#include <ucxx/request_flush.h>
#include <ucxx/request_tag.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils/callback_notifier.h>
#include <ucxx/utils/file_descriptor.h>
#include <ucxx/utils/ucx.h>
#include <ucxx/worker.h>

namespace ucxx {

Worker::Worker(std::shared_ptr<Context> context,
               const bool enableDelayedSubmission,
               const bool enableFuture)
  : _enableFuture(enableFuture)
{
  if (context == nullptr || context->getHandle() == nullptr)
    throw std::runtime_error("Context not initialized");

  ucp_worker_params_t params = {.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
                                .thread_mode = UCS_THREAD_MODE_MULTI};
  utils::ucsErrorThrow(ucp_worker_create(context->getHandle(), &params, &_handle));

  _delayedSubmissionCollection =
    std::make_shared<DelayedSubmissionCollection>(enableDelayedSubmission);

  if (context->getFeatureFlags() & UCP_FEATURE_AM) {
    unsigned int AM_MSG_ID            = 0;
    _amData                           = std::make_shared<internal::AmData>();
    _amData->_registerInflightRequest = [this](std::shared_ptr<Request> req) {
      return registerInflightRequest(req);
    };
    registerAmAllocator(UCS_MEMORY_TYPE_HOST,
                        [](size_t length) { return std::make_shared<HostBuffer>(length); });

    ucp_am_handler_param_t am_handler_param = {.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                                                             UCP_AM_HANDLER_PARAM_FIELD_CB |
                                                             UCP_AM_HANDLER_PARAM_FIELD_ARG,
                                               .id  = AM_MSG_ID,
                                               .cb  = RequestAm::recvCallback,
                                               .arg = _amData.get()};
    utils::ucsErrorThrow(ucp_worker_set_am_recv_handler(_handle, &am_handler_param));
  }

  ucxx_trace(
    "ucxx::Worker created: %p, UCP handle: %p, enableDelayedSubmission: %d, enableFuture: %d",
    this,
    _handle,
    enableDelayedSubmission,
    _enableFuture);

  setParent(std::dynamic_pointer_cast<Component>(context));
}

static void _drainCallback(void* request,
                           ucs_status_t status,
                           const ucp_tag_recv_info_t* info,
                           void* arg)
{
  *reinterpret_cast<ucs_status_t*>(request) = status;
}

void Worker::drainWorkerTagRecv()
{
  auto context = std::dynamic_pointer_cast<Context>(_parent);
  if (!(context->getFeatureFlags() & UCP_FEATURE_TAG)) return;

  ucp_tag_message_h message;
  ucp_tag_recv_info_t info;

  while ((message = ucp_tag_probe_nb(_handle, 0, 0, 1, &info)) != NULL) {
    ucxx_debug(
      "ucxx::Worker::%s, Worker: %p, UCP handle: %p, tag: 0x%lx, length: %lu, "
      "draining tag receive messages",
      __func__,
      this,
      _handle,
      info.sender_tag,
      info.length);

    std::vector<char> buf(info.length);

    ucp_request_param_t param = {
      .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE,
      .cb           = {.recv = _drainCallback},
      .datatype     = ucp_dt_make_contig(1)};

    ucs_status_ptr_t status =
      ucp_tag_msg_recv_nbx(_handle, buf.data(), info.length, message, &param);

    if (status != nullptr) {
      while (UCS_PTR_STATUS(status) == UCS_INPROGRESS)
        progress();
    }
  }
}

std::shared_ptr<RequestAm> Worker::getAmRecv(
  ucp_ep_h ep, std::function<std::shared_ptr<RequestAm>()> createAmRecvRequestFunction)
{
  std::lock_guard<std::mutex> lock(_amData->_mutex);

  auto& recvPool = _amData->_recvPool;
  auto& recvWait = _amData->_recvWait;

  auto reqs = recvPool.find(ep);
  if (reqs != recvPool.end() && !reqs->second.empty()) {
    auto req = reqs->second.front();
    reqs->second.pop();
    return req;
  } else {
    auto req        = createAmRecvRequestFunction();
    auto [queue, _] = recvWait.try_emplace(ep, std::queue<std::shared_ptr<RequestAm>>());
    queue->second.push(req);
    return req;
  }
}

std::shared_ptr<Worker> createWorker(std::shared_ptr<Context> context,
                                     const bool enableDelayedSubmission,
                                     const bool enableFuture)
{
  auto worker = std::shared_ptr<Worker>(new Worker(context, enableDelayedSubmission, enableFuture));

  // We can only get a `shared_ptr<Worker>` for the Active Messages callback after it's
  // been created, thus this cannot be in the constructor.
  if (worker->_amData != nullptr) {
    worker->_amData->_worker = worker;

    std::stringstream ownerStream;
    ownerStream << "worker " << worker->getHandle();
    worker->_amData->_ownerString = ownerStream.str();
  }

  return worker;
}

Worker::~Worker()
{
  size_t canceled = cancelInflightRequests(3000000000 /* 3s */, 3);
  ucxx_debug("ucxx::Worker::%s, Worker: %p, UCP handle: %p, canceled %lu requests",
             __func__,
             this,
             _handle,
             canceled);

  if (_progressThread.isRunning()) {
    ucxx_warn(
      "The progress thread should be explicitly stopped with `stopProgressThread()` to prevent "
      "unintended effects, such as destructors being called from that thread.");
    stopProgressThreadNoWarn();
  }
  if (_notifier && _notifier->isRunning()) {
    ucxx_warn(
      "The notifier thread should be explicitly stopped with `stopNotifierThread()` to prevent "
      "unintended effects, such as destructors being called from that thread.");
    _notifier->stopRequestNotifierThread();
  }

  drainWorkerTagRecv();

  ucp_worker_destroy(_handle);
  ucxx_trace("Worker destroyed: %p, UCP handle: %p", this, _handle);

  if (_epollFileDescriptor >= 0) close(_epollFileDescriptor);
}

ucp_worker_h Worker::getHandle() { return _handle; }

std::string Worker::getInfo()
{
  FILE* TextFileDescriptor = utils::createTextFileDescriptor();
  ucp_worker_print_info(this->_handle, TextFileDescriptor);
  return utils::decodeTextFileDescriptor(TextFileDescriptor);
}

bool Worker::isDelayedRequestSubmissionEnabled() const
{
  return _delayedSubmissionCollection->isDelayedRequestSubmissionEnabled();
}

bool Worker::isFutureEnabled() const { return _enableFuture; }

void Worker::initBlockingProgressMode()
{
  // In blocking progress mode, we create an epoll file
  // descriptor that we can wait on later.
  // We also introduce an additional eventfd to allow
  // canceling the wait.
  int err;

  // Return if blocking progress mode was already initialized
  if (_epollFileDescriptor >= 0) return;

  utils::ucsErrorThrow(ucp_worker_get_efd(_handle, &_workerFileDescriptor));

  arm();

  _epollFileDescriptor = epoll_create(1);
  if (_epollFileDescriptor == -1) throw std::ios_base::failure("epoll_create(1) returned -1");

  epoll_event workerEvent = {.events = EPOLLIN,
                             .data   = {
                                 .fd = _workerFileDescriptor,
                             }};

  err = epoll_ctl(_epollFileDescriptor, EPOLL_CTL_ADD, _workerFileDescriptor, &workerEvent);
  if (err != 0) {
    throw std::ios_base::failure(std::string("epoll_ctl() returned ") + std::to_string(err));
  }
}

int Worker::getEpollFileDescriptor()
{
  if (_epollFileDescriptor == 0)
    throw std::runtime_error("Worker not running in blocking progress mode");

  return _epollFileDescriptor;
}

bool Worker::arm()
{
  ucs_status_t status = ucp_worker_arm(_handle);
  if (status == UCS_ERR_BUSY) return false;
  utils::ucsErrorThrow(status);
  return true;
}

bool Worker::progressWorkerEvent(const int epollTimeout)
{
  int ret;
  epoll_event ev;

  if (progress()) return true;

  if ((_epollFileDescriptor == -1) || !arm()) return false;

  do {
    ret = epoll_wait(_epollFileDescriptor, &ev, 1, epollTimeout);
  } while ((ret == -1) && (errno == EINTR || errno == EAGAIN));

  return false;
}

void Worker::signal() { utils::ucsErrorThrow(ucp_worker_signal(_handle)); }

bool Worker::waitProgress()
{
  utils::ucsErrorThrow(ucp_worker_wait(_handle));
  return progress();
}

bool Worker::progressOnce() { return ucp_worker_progress(_handle) != 0; }

bool Worker::progressPending()
{
  bool ret = false, prog = false;
  do {
    prog = progressOnce();
    ret |= prog;
  } while (prog);
  return ret;
}

bool Worker::progress()
{
  bool ret                     = progressPending();
  bool progressScheduledCancel = false;

  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);

    // Before canceling requests scheduled for cancelation, attempt to let them complete.
    progressScheduledCancel =
      _inflightRequestsToCancel != nullptr && _inflightRequestsToCancel->size() > 0;
  }
  if (progressScheduledCancel) ret |= progressPending();

  // Requests that were not completed now must be canceled.
  if (cancelInflightRequests(3000000000 /* 3s */, 3) > 0) ret |= progressPending();

  return ret;
}

void Worker::registerDelayedSubmission(std::shared_ptr<Request> request,
                                       DelayedSubmissionCallbackType callback)
{
  if (_delayedSubmissionCollection->isDelayedRequestSubmissionEnabled()) {
    _delayedSubmissionCollection->registerRequest(request, callback);

    /* Waking the progress event is needed here because the UCX request is
     * not dispatched immediately. Thus we must signal the progress task so
     * it will ensure the request is dispatched.
     */
    signal();
  } else {
    callback();
  }
}

bool Worker::registerGenericPre(DelayedSubmissionCallbackType callback, uint64_t period)
{
  if (std::this_thread::get_id() == getProgressThreadId()) {
    /**
     * If the method is called from within the progress thread (e.g., from the
     * listener callback), execute it immediately.
     */
    callback();

    return true;
  } else {
    utils::CallbackNotifier callbackNotifier{};
    auto notifiableCallback = [&callback, &callbackNotifier]() {
      callback();
      callbackNotifier.set();
    };

    auto id = _delayedSubmissionCollection->registerGenericPre(notifiableCallback);

    /* Waking the progress event is needed here because the UCX request is
     * not dispatched immediately. Thus we must signal the progress task so
     * it will ensure the request is dispatched.
     */
    std::function<void()> signalWorkerFunction = []() {};
    if (_progressThread.isRunning() && !_progressThread.pollingMode()) {
      signalWorkerFunction = [this]() { return this->signal(); };
    }
    signalWorkerFunction();

    size_t retryCount = 0;
    while (true) {
      auto ret = callbackNotifier.wait(period, signalWorkerFunction);

      try {
        if (!ret) _delayedSubmissionCollection->cancelGenericPre(id);
        return ret;
      } catch (const std::runtime_error& e) {
        if (++retryCount % 10 == 0)
          ucxx_warn(
            "Could not cancel after %lu attempts, the callback has not returned and the process "
            "may stop responding.",
            retryCount);
      }
    }
  }
}

bool Worker::registerGenericPost(DelayedSubmissionCallbackType callback, uint64_t period)
{
  if (std::this_thread::get_id() == getProgressThreadId()) {
    /**
     * If the method is called from within the progress thread (e.g., from the
     * listener callback), execute it immediately.
     */
    callback();

    return true;
  } else {
    utils::CallbackNotifier callbackNotifier{};
    auto notifiableCallback = [&callback, &callbackNotifier]() {
      callback();
      callbackNotifier.set();
    };

    auto id = _delayedSubmissionCollection->registerGenericPost(notifiableCallback);

    /* Waking the progress event is needed here because the UCX request is
     * not dispatched immediately. Thus we must signal the progress task so
     * it will ensure the request is dispatched.
     */
    std::function<void()> signalWorkerFunction = []() {};
    if (_progressThread.isRunning() && !_progressThread.pollingMode()) {
      signalWorkerFunction = [this]() { return this->signal(); };
    }
    signalWorkerFunction();

    size_t retryCount = 0;
    while (true) {
      auto ret = callbackNotifier.wait(period, signalWorkerFunction);

      try {
        if (!ret) _delayedSubmissionCollection->cancelGenericPost(id);
        return ret;
      } catch (const std::runtime_error& e) {
        if (++retryCount % 10 == 0)
          ucxx_warn(
            "Could not cancel after %lu attempts, the callback has not returned and the process "
            "may stop responding.",
            retryCount);
      }
    }
  }
}

#define THROW_FUTURE_NOT_IMPLEMENTED()                                                      \
  do {                                                                                      \
    throw std::runtime_error(                                                               \
      "ucxx::Worker's future support not implemented, please ensure you use an "            \
      "implementation with future support and that enableFuture=true is set when creating " \
      "the Worker to use this method.");                                                    \
  } while (0)

void Worker::populateFuturesPool() { THROW_FUTURE_NOT_IMPLEMENTED(); }

void Worker::clearFuturesPool() { THROW_FUTURE_NOT_IMPLEMENTED(); }

std::shared_ptr<Future> Worker::getFuture() { THROW_FUTURE_NOT_IMPLEMENTED(); }

RequestNotifierWaitState Worker::waitRequestNotifier(uint64_t periodNs)
{
  THROW_FUTURE_NOT_IMPLEMENTED();
}

void Worker::runRequestNotifier() { THROW_FUTURE_NOT_IMPLEMENTED(); }

void Worker::stopRequestNotifierThread() { THROW_FUTURE_NOT_IMPLEMENTED(); }

void Worker::setProgressThreadStartCallback(std::function<void(void*)> callback, void* callbackArg)
{
  _progressThreadStartCallback    = callback;
  _progressThreadStartCallbackArg = callbackArg;
}

void Worker::startProgressThread(const bool pollingMode, const int epollTimeout)
{
  if (_progressThread.isRunning()) {
    ucxx_debug(
      "ucxx::Worker::%s, Worker: %p, UCP handle: %p, worker progress thread "
      "already running",
      __func__,
      this,
      _handle);
    return;
  }

  std::function<bool()> progressFunction;
  std::function<void()> signalWorkerFunction;
  if (pollingMode) {
    progressFunction     = [this]() { return this->progress(); };
    signalWorkerFunction = []() {};
  } else {
    initBlockingProgressMode();
    progressFunction = [this, epollTimeout]() { return this->progressWorkerEvent(epollTimeout); };
    signalWorkerFunction = [this]() { return this->signal(); };
  }

  auto setThreadId = [this]() { _progressThreadId = std::this_thread::get_id(); };

  _progressThread = WorkerProgressThread(pollingMode,
                                         progressFunction,
                                         signalWorkerFunction,
                                         setThreadId,
                                         _progressThreadStartCallback,
                                         _progressThreadStartCallbackArg,
                                         _delayedSubmissionCollection);
}

void Worker::stopProgressThreadNoWarn() { _progressThread.stop(); }

void Worker::stopProgressThread()
{
  if (!_progressThread.isRunning())
    ucxx_debug(
      "ucxx::Worker::%s, Worker: %p, UCP handle: %p, worker progress thread not "
      "running or already stopped",
      __func__,
      this,
      _handle);
  else
    stopProgressThreadNoWarn();
}

bool Worker::isProgressThreadRunning() { return _progressThread.isRunning(); }

std::thread::id Worker::getProgressThreadId() { return _progressThreadId; }

size_t Worker::cancelInflightRequests(uint64_t period, uint64_t maxAttempts)
{
  size_t canceled = 0;

  auto inflightRequestsToCancel = std::make_unique<InflightRequests>();
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    std::swap(_inflightRequestsToCancel, inflightRequestsToCancel);
  }

  if (std::this_thread::get_id() == getProgressThreadId()) {
    canceled = inflightRequestsToCancel->cancelAll();
    for (uint64_t i = 0; i < maxAttempts && inflightRequestsToCancel->getCancelingSize() > 0; ++i)
      progressPending();
  } else if (isProgressThreadRunning()) {
    bool cancelSuccess = false;
    for (uint64_t i = 0; i < maxAttempts && !cancelSuccess; ++i) {
      if (!registerGenericPre(
            [&canceled, &inflightRequestsToCancel]() {
              canceled += inflightRequestsToCancel->cancelAll();
            },
            period))
        continue;

      if (!registerGenericPost(
            [this, &inflightRequestsToCancel, &cancelSuccess]() {
              cancelSuccess = inflightRequestsToCancel->getCancelingSize() == 0;
            },
            period))
        continue;
    }

    if (!cancelSuccess)
      ucxx_debug(
        "ucxx::Worker::%s, Worker: %p, UCP handle: %p, all attempts to cancel "
        "inflight requests failed",
        __func__,
        this,
        _handle);
  } else {
    canceled = inflightRequestsToCancel->cancelAll();
  }

  if (inflightRequestsToCancel->getCancelingSize() > 0) {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    _inflightRequestsToCancel->merge(inflightRequestsToCancel->release());
  }

  return canceled;
}

void Worker::scheduleRequestCancel(TrackedRequestsPtr trackedRequests)
{
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    ucxx_debug(
      "ucxx::Worker::%s, Worker: %p, UCP handle: %p, scheduling cancelation of "
      "%lu requests",
      __func__,
      this,
      _handle,
      trackedRequests->_inflight.size() + trackedRequests->_canceling.size());
    _inflightRequestsToCancel->merge(std::move(trackedRequests));
  }
}

std::shared_ptr<Request> Worker::registerInflightRequest(std::shared_ptr<Request> request)
{
  if (!request->isCompleted()) {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    _inflightRequests->insert(request);
  }

  return request;
}

void Worker::removeInflightRequest(const Request* const request)
{
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    _inflightRequests->remove(request);
  }
}

bool Worker::tagProbe(const Tag tag)
{
  if (!isProgressThreadRunning()) {
    progress();
  } else {
    /**
     * To ensure the worker was progressed at least once, we must make sure a callback runs
     * pre-progressing, and another one runs post-progress. Running post-progress only may
     * indicate the progress thread has immediately finished executing and post-progress
     * ran without a further progress operation.
     */
    std::ignore = registerGenericPre([]() {}, 3000000000 /* 3s */);
    std::ignore = registerGenericPost([]() {}, 3000000000 /* 3s */);
  }

  ucp_tag_recv_info_t info;
  ucp_tag_message_h tag_message = ucp_tag_probe_nb(_handle, tag, -1, 0, &info);

  return tag_message != NULL;
}

std::shared_ptr<Request> Worker::tagRecv(void* buffer,
                                         size_t length,
                                         Tag tag,
                                         TagMask tagMask,
                                         const bool enableFuture,
                                         RequestCallbackUserFunction callbackFunction,
                                         RequestCallbackUserData callbackData)
{
  auto worker = std::dynamic_pointer_cast<Worker>(shared_from_this());
  return registerInflightRequest(createRequestTag(worker,
                                                  data::TagReceive(buffer, length, tag, tagMask),
                                                  enableFuture,
                                                  callbackFunction,
                                                  callbackData));
}

std::shared_ptr<Address> Worker::getAddress()
{
  auto worker  = std::dynamic_pointer_cast<Worker>(shared_from_this());
  auto address = ucxx::createAddressFromWorker(worker);
  return address;
}

std::shared_ptr<Endpoint> Worker::createEndpointFromHostname(std::string ipAddress,
                                                             uint16_t port,
                                                             bool endpointErrorHandling)
{
  auto worker   = std::dynamic_pointer_cast<Worker>(shared_from_this());
  auto endpoint = ucxx::createEndpointFromHostname(worker, ipAddress, port, endpointErrorHandling);
  return endpoint;
}

std::shared_ptr<Endpoint> Worker::createEndpointFromWorkerAddress(std::shared_ptr<Address> address,
                                                                  bool endpointErrorHandling)
{
  auto worker   = std::dynamic_pointer_cast<Worker>(shared_from_this());
  auto endpoint = ucxx::createEndpointFromWorkerAddress(worker, address, endpointErrorHandling);
  return endpoint;
}

std::shared_ptr<Listener> Worker::createListener(uint16_t port,
                                                 ucp_listener_conn_callback_t callback,
                                                 void* callbackArgs)
{
  auto worker   = std::dynamic_pointer_cast<Worker>(shared_from_this());
  auto listener = ucxx::createListener(worker, port, callback, callbackArgs);
  return listener;
}

void Worker::registerAmAllocator(ucs_memory_type_t memoryType, AmAllocatorType allocator)
{
  if (_amData == nullptr)
    throw std::runtime_error("Active Messages was not enabled during context creation");
  _amData->_allocators.insert_or_assign(memoryType, allocator);
}

void Worker::registerAmReceiverCallback(AmReceiverCallbackInfo info,
                                        AmReceiverCallbackType callback)
{
  if (info.owner == "ucxx") throw std::runtime_error("The owner name 'ucxx' is reserved.");
  if (_amData->_receiverCallbacks.find(info.owner) == _amData->_receiverCallbacks.end())
    _amData->_receiverCallbacks[info.owner] = {};
  if (_amData->_receiverCallbacks[info.owner].find(info.id) !=
      _amData->_receiverCallbacks[info.owner].end())
    throw std::runtime_error("Callback with given owner and identifier is already registered");

  _amData->_receiverCallbacks[info.owner][info.id] = callback;
}

bool Worker::amProbe(const ucp_ep_h endpointHandle) const
{
  return _amData->_recvPool.find(endpointHandle) != _amData->_recvPool.end();
}

std::shared_ptr<Request> Worker::flush(const bool enableFuture,
                                       RequestCallbackUserFunction callbackFunction,
                                       RequestCallbackUserData callbackData)
{
  auto worker = std::dynamic_pointer_cast<Worker>(shared_from_this());
  return registerInflightRequest(
    createRequestFlush(worker, data::Flush(), enableFuture, callbackFunction, callbackData));
}

}  // namespace ucxx
