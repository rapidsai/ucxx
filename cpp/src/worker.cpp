/**
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <functional>
#include <ios>
#include <mutex>
#include <queue>

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <ucxx/request_tag.h>
#include <ucxx/utils/file_descriptor.h>
#include <ucxx/utils/ucx.h>
#include <ucxx/worker.h>

namespace ucxx {

Worker::Worker(std::shared_ptr<Context> context, const bool enableDelayedSubmission)
{
  ucp_worker_params_t params{};

  if (context == nullptr || context->getHandle() == nullptr)
    throw std::runtime_error("Context not initialized");

  params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  params.thread_mode = UCS_THREAD_MODE_MULTI;
  utils::ucsErrorThrow(ucp_worker_create(context->getHandle(), &params, &_handle));

  if (enableDelayedSubmission)
    _delayedSubmissionCollection = std::make_shared<DelayedSubmissionCollection>();

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
  *(ucs_status_t*)request = status;
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

std::shared_ptr<Worker> createWorker(std::shared_ptr<Context> context,
                                     const bool enableDelayedSubmission)
{
  return std::shared_ptr<Worker>(new Worker(context, enableDelayedSubmission));
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

bool Worker::progressWorkerEvent()
{
  int ret;
  epoll_event ev;

  cancelInflightRequests();

  if (progress()) return true;

  if ((_epollFileDescriptor == -1) || !arm()) return false;

  do {
    ret = epoll_wait(_epollFileDescriptor, &ev, 1, -1);
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

  // Before canceling requests scheduled for cancellation, attempt to let them complete.
  if (_inflightRequestsToCancel > 0) ret |= progressPending();

  // Requests that were not completed now must be canceled.
  if (cancelInflightRequests() > 0) ret |= progressPending();

  return ret;
}

void Worker::registerDelayedSubmission(DelayedSubmissionCallbackType callback)
{
  if (_delayedSubmissionCollection == nullptr) {
    callback();
  } else {
    _delayedSubmissionCollection->registerRequest(callback);

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

void Worker::startProgressThread(const bool pollingMode)
{
  if (_progressThread) {
    ucxx_warn("Worker progress thread already running");
    return;
  }

  if (!pollingMode) initBlockingProgressMode();
  auto progressFunction = pollingMode ? std::bind(&Worker::progress, this)
                                      : std::bind(&Worker::progressWorkerEvent, this);

  _progressThread = std::make_shared<WorkerProgressThread>(pollingMode,
                                                           progressFunction,
                                                           _progressThreadStartCallback,
                                                           _progressThreadStartCallbackArg,
                                                           _delayedSubmissionCollection);
}

void Worker::stopProgressThreadNoWarn()
{
  if (_progressThread && !_progressThread->pollingMode()) signal();
  _progressThread = nullptr;
}

void Worker::stopProgressThread()
{
  if (!_progressThread)
    ucxx_warn("Worker progress thread not running or already stopped");
  else
    stopProgressThreadNoWarn();
}

size_t Worker::cancelInflightRequests()
{
  auto inflightRequestsToCancel = std::make_shared<InflightRequests>();
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    std::swap(_inflightRequestsToCancel, inflightRequestsToCancel);
  }
  return inflightRequestsToCancel->cancelAll();
}

void Worker::scheduleRequestCancel(std::shared_ptr<InflightRequests> inflightRequests)
{
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    ucxx_debug("Scheduling cancellation of %lu requests", inflightRequests->size());
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

bool Worker::tagProbe(ucp_tag_t tag)
{
  ucp_tag_recv_info_t info;
  ucp_tag_message_h tag_message = ucp_tag_probe_nb(_handle, tag, -1, 0, &info);

  return tag_message != NULL;
}

std::shared_ptr<Request> Worker::tagRecv(
  void* buffer,
  size_t length,
  ucp_tag_t tag,
  const bool enableFuture,
  std::function<void(std::shared_ptr<void>)> callbackFunction,
  std::shared_ptr<void> callbackData)
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

}  // namespace ucxx
