/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/constructors.h>
#include <ucxx/context.h>
#include <ucxx/delayed_notification_request.h>
#include <ucxx/notifier.h>
#include <ucxx/transfer_common.h>
#include <ucxx/utils.h>

#ifdef UCXX_ENABLE_PYTHON
#include <ucxx/python/python_future.h>
#endif

namespace ucxx {

class UCXXAddress;
class UCXXEndpoint;
class UCXXListener;

class UCXXWorker : public UCXXComponent {
 private:
  ucp_worker_h _handle{nullptr};
  int _epoll_fd{-1};
  int _worker_fd{-1};
  int _cancel_efd{-1};
  std::thread _progressThread{};
  bool _stopProgressThread{false};
  bool _progressThreadPollingMode{false};
  inflight_requests_t _inflightRequestsToCancel{std::make_shared<inflight_request_map_t>()};
  std::mutex _inflightMutex{};
  std::function<void(void*)> _progressThreadStartCallback{nullptr};
  void* _progressThreadStartCallbackArg{nullptr};
  DelayedNotificationRequestCollection _delayedNotificationRequestCollection{};
  std::mutex _pythonFuturesPoolMutex{};
  std::queue<std::shared_ptr<PythonFuture>> _pythonFuturesPool{};
  std::shared_ptr<UCXXNotifier> _notifier{std::make_shared<UCXXNotifier>()};

  UCXXWorker(std::shared_ptr<ucxx::UCXXContext> context)
  {
    ucp_worker_params_t worker_params;

    if (context == nullptr || context->get_handle() == nullptr)
      throw std::runtime_error("UCXXContext not initialized");

    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    assert_ucs_status(ucp_worker_create(context->get_handle(), &worker_params, &_handle));

    setParent(std::dynamic_pointer_cast<UCXXComponent>(context));
  }

  void drainWorkerTagRecv()
  {
    auto context = std::dynamic_pointer_cast<UCXXContext>(_parent);
    if (!(context->get_feature_flags() & UCP_FEATURE_TAG)) return;

    ucp_tag_message_h message;
    ucp_tag_recv_info_t info;

    while ((message = ucp_tag_probe_nb(_handle, 0, 0, 1, &info)) != NULL) {
      ucxx_debug("Draining tag receive messages, worker: %p, tag: 0x%lx, length: %lu",
                 _handle,
                 info.sender_tag,
                 info.length);

      std::shared_ptr<ucxx_request_t> request = std::make_shared<ucxx_request_t>();
      ucp_request_param_t param               = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                                   UCP_OP_ATTR_FIELD_DATATYPE |
                                                   UCP_OP_ATTR_FIELD_USER_DATA,
                                   .datatype  = ucp_dt_make_contig(1),
                                   .user_data = request.get()};

      std::unique_ptr<char> buf = std::make_unique<char>(info.length);
      ucs_status_ptr_t status =
        ucp_tag_msg_recv_nbx(_handle, buf.get(), info.length, message, &param);
      request_wait(_handle, status, request.get(), "drain_tag_recv");

      while (request->status == UCS_INPROGRESS)
        progress();
    }
  }

 public:
  UCXXWorker() = delete;

  UCXXWorker(const UCXXWorker&) = delete;
  UCXXWorker& operator=(UCXXWorker const&) = delete;

  UCXXWorker(UCXXWorker&& o) = delete;
  UCXXWorker& operator=(UCXXWorker&& o) = delete;

  template <class... Args>
  friend std::shared_ptr<UCXXWorker> createWorker(Args&&... args)
  {
    return std::shared_ptr<UCXXWorker>(new UCXXWorker(std::forward<Args>(args)...));
  }

  ~UCXXWorker()
  {
    stopProgressThread();
    _notifier->stopRequestNotifierThread();

    drainWorkerTagRecv();

    ucp_worker_destroy(_handle);

    if (_epoll_fd >= 0) close(_epoll_fd);
    if (_cancel_efd >= 0) close(_cancel_efd);
  }

  ucp_worker_h get_handle() { return _handle; }

  void init_blocking_progress_mode()
  {
    // In blocking progress mode, we create an epoll file
    // descriptor that we can wait on later.
    // We also introduce an additional eventfd to allow
    // canceling the wait.
    int err;

    // Return if blocking progress mode was already initialized
    if (_epoll_fd >= 0) return;

    assert_ucs_status(ucp_worker_get_efd(_handle, &_worker_fd));

    arm();

    _epoll_fd = epoll_create(1);
    if (_epoll_fd == -1) throw std::ios_base::failure("epoll_create(1) returned -1");

    _cancel_efd = eventfd(0, EFD_NONBLOCK);
    if (_cancel_efd < 0) throw std::ios_base::failure("eventfd(0, EFD_NONBLOCK) returned -1");

    epoll_event worker_ev = {.events = EPOLLIN,
                             .data   = {
                               .fd = _worker_fd,
                             }};
    epoll_event cancel_ev = {.events = EPOLLIN,
                             .data   = {
                               .fd = _cancel_efd,
                             }};

    err = epoll_ctl(_epoll_fd, EPOLL_CTL_ADD, _worker_fd, &worker_ev);
    if (err != 0) throw std::ios_base::failure(std::string("epoll_ctl() returned " + err));
    err = epoll_ctl(_epoll_fd, EPOLL_CTL_ADD, _cancel_efd, &cancel_ev);
    if (err != 0) throw std::ios_base::failure(std::string("epoll_ctl() returned " + err));
  }

  bool arm()
  {
    ucs_status_t status = ucp_worker_arm(_handle);
    if (status == UCS_ERR_BUSY) return false;
    assert_ucs_status(status);
    return true;
  }

  bool progress_worker_event()
  {
    int ret;
    epoll_event ev;

    if (progress_once()) return true;

    if ((_epoll_fd == -1) || !arm()) return false;

    do {
      ret = epoll_wait(_epoll_fd, &ev, 1, -1);
    } while ((ret == -1) && (errno == EINTR || errno == EAGAIN));

    return false;
  }

  void cancel_progress_worker_event()
  {
    int err = eventfd_write(_cancel_efd, 1);
    if (err < 0) throw std::ios_base::failure(std::string("eventfd_write() returned " + err));
  }

  bool wait_progress()
  {
    assert_ucs_status(ucp_worker_wait(_handle));
    return progress_once();
  }

  bool progress_once() { return ucp_worker_progress(_handle) != 0; }

  void progress()
  {
    while (ucp_worker_progress(_handle) != 0)
      ;
  }

  void setProgressThreadStartCallback(std::function<void(void*)> callback, void* callbackArg)
  {
    _progressThreadStartCallback    = callback;
    _progressThreadStartCallbackArg = callbackArg;
  }

  static void progressUntilSync(
    std::function<bool(void)> progressFunction,
    const bool& stopProgressThread,
    std::function<void(void*)> progressThreadStartCallback,
    void* progressThreadStartCallbackArg,
    DelayedNotificationRequestCollection& delayedNotificationRequestCollection)
  {
    if (progressThreadStartCallback) progressThreadStartCallback(progressThreadStartCallbackArg);

    while (!stopProgressThread) {
      delayedNotificationRequestCollection.process();

      progressFunction();
    }
  }

  void registerDelayedNotificationRequest(std::function<void(std::shared_ptr<void>)> callback,
                                          std::shared_ptr<void> callbackData)
  {
    _delayedNotificationRequestCollection.registerRequest(callback, callbackData);
  }

  void populatePythonFuturesPool()
  {
    ucxx_trace_req("populatePythonFuturesPool: %p %p", this, shared_from_this().get());
    // If the pool goes under half expected size, fill it up again.
    if (_pythonFuturesPool.size() < 50) {
      std::lock_guard<std::mutex> lock(_pythonFuturesPoolMutex);
      while (_pythonFuturesPool.size() < 100)
        _pythonFuturesPool.emplace(std::make_shared<PythonFuture>(_notifier));
    }
  }

  std::shared_ptr<PythonFuture> getPythonFuture()
  {
    std::shared_ptr<PythonFuture> ret{nullptr};
    {
      std::lock_guard<std::mutex> lock(_pythonFuturesPoolMutex);
      ret = _pythonFuturesPool.front();
      _pythonFuturesPool.pop();
    }
    ucxx_trace_req("getPythonFuture: %p %p", ret.get(), ret->getHandle());
    return ret;
  }

  bool waitRequestNotifier() { return _notifier->waitRequestNotifier(); }

  void runRequestNotifier() { return _notifier->runRequestNotifier(); }

  void stopRequestNotifierThread() { return _notifier->stopRequestNotifierThread(); }

  void startProgressThread(const bool pollingMode = true)
  {
    if (_progressThread.joinable()) return;

    _stopProgressThread        = false;
    _progressThreadPollingMode = pollingMode;

    if (pollingMode) init_blocking_progress_mode();
    auto progressFunction = pollingMode ? std::bind(&UCXXWorker::progress_worker_event, this)
                                        : std::bind(&UCXXWorker::progress_once, this);

    _progressThread = std::thread(UCXXWorker::progressUntilSync,
                                  progressFunction,
                                  std::ref(_stopProgressThread),
                                  _progressThreadStartCallback,
                                  _progressThreadStartCallbackArg,
                                  std::ref(_delayedNotificationRequestCollection));
  }

  void stopProgressThread()
  {
    if (!_progressThread.joinable()) return;

    _stopProgressThread = true;
    if (_progressThreadPollingMode) { cancel_progress_worker_event(); }
    _progressThread.join();
  }

  inline size_t cancelInflightRequests();

  void scheduleRequestCancel(inflight_requests_t inflightRequests)
  {
    std::lock_guard<std::mutex> lock(_inflightMutex);
    _inflightRequestsToCancel->insert(inflightRequests->begin(), inflightRequests->end());
  }

  bool tagProbe(ucp_tag_t tag)
  {
    ucp_tag_recv_info_t info;
    ucp_tag_message_h tag_message = ucp_tag_probe_nb(_handle, tag, -1, 0, &info);

    return tag_message != NULL;
  }

  std::shared_ptr<UCXXAddress> getAddress()
  {
    auto worker  = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
    auto address = ucxx::createAddressFromWorker(worker);
    return address;
  }

  std::shared_ptr<UCXXEndpoint> createEndpointFromHostname(std::string ip_address,
                                                           uint16_t port                = 0,
                                                           bool endpoint_error_handling = true)
  {
    auto worker = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
    auto endpoint =
      ucxx::createEndpointFromHostname(worker, ip_address, port, endpoint_error_handling);
    return endpoint;
  }

  std::shared_ptr<UCXXEndpoint> createEndpointFromWorkerAddress(
    std::shared_ptr<UCXXAddress> address, bool endpoint_error_handling = true)
  {
    auto worker   = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
    auto endpoint = ucxx::createEndpointFromWorkerAddress(worker, address, endpoint_error_handling);
    return endpoint;
  }

  std::shared_ptr<UCXXListener> createListener(uint16_t port,
                                               ucp_listener_conn_callback_t callback,
                                               void* callback_args)
  {
    auto worker   = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
    auto listener = ucxx::createListener(worker, port, callback, callback_args);
    return listener;
  }
};

}  // namespace ucxx
