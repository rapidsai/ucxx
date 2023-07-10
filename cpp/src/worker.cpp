/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <condition_variable>
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
#include <ucxx/request_tag.h>
#include <ucxx/utils/condition.h>
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
      this->registerInflightRequest(req);
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

  ucxx_trace("Worker created: %p, enableDelayedSubmission: %d, enableFuture: %d",
             this,
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
    ucxx_debug("Draining tag receive messages, worker: %p, tag: 0x%lx, length: %lu",
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
  size_t canceled = cancelInflightRequests();
  ucxx_debug("Worker %p canceled %lu requests", _handle, canceled);

  stopProgressThreadNoWarn();
  if (_notifier) _notifier->stopRequestNotifierThread();

  drainWorkerTagRecv();

  ucp_worker_destroy(_handle);
  ucxx_trace("Worker destroyed: %p", _handle);

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
  if (err != 0) throw std::ios_base::failure(std::string("epoll_ctl() returned " + err));
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

  cancelInflightRequests();

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
  bool ret = progressPending();

  // Before canceling requests scheduled for cancelation, attempt to let them complete.
  if (_inflightRequestsToCancel > 0) ret |= progressPending();

  // Requests that were not completed now must be canceled.
  if (cancelInflightRequests() > 0) ret |= progressPending();

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

void Worker::registerGenericPre(DelayedSubmissionCallbackType callback)
{
  if (_progressThread != nullptr && std::this_thread::get_id() == _progressThread->getId()) {
    /**
     * If the method is called from within the progress thread (e.g., from the
     * listener callback), execute it immediately.
     */
    callback();
  } else {
    _delayedSubmissionCollection->registerGenericPre(callback);

    /* Waking the progress event is needed here because the UCX request is
     * not dispatched immediately. Thus we must signal the progress task so
     * it will ensure the request is dispatched.
     */
    signal();
  }
}

void Worker::registerGenericPost(DelayedSubmissionCallbackType callback)
{
  if (_progressThread != nullptr && std::this_thread::get_id() == _progressThread->getId()) {
    /**
     * If the method is called from within the progress thread (e.g., from the
     * listener callback), execute it immediately.
     */
    callback();
  } else {
    _delayedSubmissionCollection->registerGenericPost(callback);

    /* Waking the progress event is needed here because the UCX request is
     * not dispatched immediately. Thus we must signal the progress task so
     * it will ensure the request is dispatched.
     */
    signal();
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
  if (_progressThread) {
    ucxx_warn("Worker progress thread already running");
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

  _progressThread = std::make_shared<WorkerProgressThread>(pollingMode,
                                                           progressFunction,
                                                           signalWorkerFunction,
                                                           _progressThreadStartCallback,
                                                           _progressThreadStartCallbackArg,
                                                           _delayedSubmissionCollection);
}

void Worker::stopProgressThreadNoWarn() { _progressThread = nullptr; }

void Worker::stopProgressThread()
{
  if (!_progressThread)
    ucxx_warn("Worker progress thread not running or already stopped");
  else
    stopProgressThreadNoWarn();
}

bool Worker::isProgressThreadRunning() { return _progressThread != nullptr; }

size_t Worker::cancelInflightRequests()
{
  size_t canceled = 0;

  auto inflightRequestsToCancel = std::make_shared<InflightRequests>();
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    std::swap(_inflightRequestsToCancel, inflightRequestsToCancel);
  }

  if (isProgressThreadRunning()) {
    auto statusMutex             = std::make_shared<std::mutex>();
    auto statusConditionVariable = std::make_shared<std::condition_variable>();
    auto pre                     = std::make_shared<bool>(false);
    auto post                    = std::make_shared<bool>(false);

    auto setterPre = [this, &canceled, pre]() {
      canceled = _inflightRequests->cancelAll();
      *pre     = true;
    };
    auto getterPre = [pre]() { return *pre; };

    registerGenericPre([&statusMutex, &statusConditionVariable, &setterPre]() {
      ucxx::utils::conditionSetter(statusMutex, statusConditionVariable, setterPre);
    });
    ucxx::utils::conditionGetter(statusMutex, statusConditionVariable, pre, getterPre);

    auto setterPost = [this, post]() { *post = true; };
    auto getterPost = [post]() { return *post; };

    registerGenericPost([&statusMutex, &statusConditionVariable, &setterPost]() {
      ucxx::utils::conditionSetter(statusMutex, statusConditionVariable, setterPost);
    });
    ucxx::utils::conditionGetter(statusMutex, statusConditionVariable, post, getterPost);
  } else {
    canceled = inflightRequestsToCancel->cancelAll();
  }

  return canceled;
}

void Worker::scheduleRequestCancel(std::shared_ptr<InflightRequests> inflightRequests)
{
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    ucxx_debug("Scheduling cancelation of %lu requests", inflightRequests->size());
    _inflightRequestsToCancel->merge(inflightRequests->release());
  }
}

void Worker::registerInflightRequest(std::shared_ptr<Request> request)
{
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    _inflightRequests->insert(request);
  }
}

void Worker::removeInflightRequest(const Request* const request)
{
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    _inflightRequests->remove(request);
  }
}

bool Worker::tagProbe(const ucp_tag_t tag)
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
    auto statusMutex             = std::make_shared<std::mutex>();
    auto statusConditionVariable = std::make_shared<std::condition_variable>();
    auto pre                     = std::make_shared<bool>(false);
    auto post                    = std::make_shared<bool>(false);

    auto setterPre = [this, pre]() { *pre = true; };
    auto getterPre = [pre]() { return *pre; };

    registerGenericPre([&statusMutex, &statusConditionVariable, &setterPre]() {
      ucxx::utils::conditionSetter(statusMutex, statusConditionVariable, setterPre);
    });
    ucxx::utils::conditionGetter(statusMutex, statusConditionVariable, pre, getterPre);

    auto setterPost = [this, post]() { *post = true; };
    auto getterPost = [post]() { return *post; };

    registerGenericPost([&statusMutex, &statusConditionVariable, &setterPost]() {
      ucxx::utils::conditionSetter(statusMutex, statusConditionVariable, setterPost);
    });
    ucxx::utils::conditionGetter(statusMutex, statusConditionVariable, post, getterPost);
  }

  ucp_tag_recv_info_t info;
  ucp_tag_message_h tag_message = ucp_tag_probe_nb(_handle, tag, -1, 0, &info);

  return tag_message != NULL;
}

std::shared_ptr<Request> Worker::tagRecv(void* buffer,
                                         size_t length,
                                         ucp_tag_t tag,
                                         const bool enableFuture,
                                         RequestCallbackUserFunction callbackFunction,
                                         RequestCallbackUserData callbackData)
{
  auto worker  = std::dynamic_pointer_cast<Worker>(shared_from_this());
  auto request = createRequestTag(
    worker, false, buffer, length, tag, enableFuture, callbackFunction, callbackData);
  registerInflightRequest(request);
  return request;
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
    throw std::runtime_error("Active Messages wasn not enabled during context creation");
  _amData->_allocators.insert_or_assign(memoryType, allocator);
}

bool Worker::amProbe(const ucp_ep_h endpointHandle) const
{
  return _amData->_recvPool.find(endpointHandle) != _amData->_recvPool.end();
}

}  // namespace ucxx
