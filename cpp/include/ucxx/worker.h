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

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/constructors.h>
#include <ucxx/context.h>
#include <ucxx/notification_request.h>
#include <ucxx/worker_progress_thread.h>

#if UCXX_ENABLE_PYTHON
#include <ucxx/python/notifier.h>
#include <ucxx/python/python_future.h>
#endif

namespace ucxx {

class UCXXAddress;
class UCXXEndpoint;
class UCXXListener;

class UCXXWorker : public UCXXComponent {
 private:
  ucp_worker_h _handle{nullptr};
  int _epollFileDescriptor{-1};
  int _workerFileDescriptor{-1};
  int _wakeFileDescriptor{-1};
  std::shared_ptr<UCXXWorkerProgressThread> _progressThread{nullptr};
  inflight_requests_t _inflightRequestsToCancel{std::make_shared<inflight_request_map_t>()};
  std::mutex _inflightMutex{};
  std::function<void(void*)> _progressThreadStartCallback{nullptr};
  void* _progressThreadStartCallbackArg{nullptr};
  std::shared_ptr<DelayedNotificationRequestCollection> _delayedNotificationRequestCollection{
    nullptr};
  std::mutex _pythonFuturesPoolMutex{};
#if UCXX_ENABLE_PYTHON
  std::queue<std::shared_ptr<PythonFuture>> _pythonFuturesPool{};
  std::shared_ptr<UCXXNotifier> _notifier{createNotifier()};
#endif

  UCXXWorker(std::shared_ptr<UCXXContext> context, const bool enableDelayedNotification = false);

  void drainWorkerTagRecv();

 public:
  UCXXWorker()                  = delete;
  UCXXWorker(const UCXXWorker&) = delete;
  UCXXWorker& operator=(UCXXWorker const&) = delete;
  UCXXWorker(UCXXWorker&& o)               = delete;
  UCXXWorker& operator=(UCXXWorker&& o) = delete;

  friend std::shared_ptr<UCXXWorker> createWorker(std::shared_ptr<UCXXContext> context,
                                                  const bool enableDelayedNotification);

  ~UCXXWorker();

  ucp_worker_h get_handle();

  void init_blocking_progress_mode();

  bool arm();

  bool progress_worker_event();

  void wakeProgressEvent();

  bool wait_progress();

  bool progress_once();

  void progress();

  void registerNotificationRequest(NotificationRequestCallbackType callback);

  void populatePythonFuturesPool();

  std::shared_ptr<PythonFuture> getPythonFuture();

  bool waitRequestNotifier();

  void runRequestNotifier();

  void stopRequestNotifierThread();

  void setProgressThreadStartCallback(std::function<void(void*)> callback, void* callbackArg);

  void startProgressThread(const bool pollingMode = true);

  void stopProgressThread();

  inline size_t cancelInflightRequests();

  void scheduleRequestCancel(inflight_requests_t inflightRequests);

  bool tagProbe(ucp_tag_t tag);

  std::shared_ptr<UCXXAddress> getAddress();

  std::shared_ptr<UCXXEndpoint> createEndpointFromHostname(std::string ip_address,
                                                           uint16_t port                = 0,
                                                           bool endpoint_error_handling = true);

  std::shared_ptr<UCXXEndpoint> createEndpointFromWorkerAddress(
    std::shared_ptr<UCXXAddress> address, bool endpoint_error_handling = true);

  std::shared_ptr<UCXXListener> createListener(uint16_t port,
                                               ucp_listener_conn_callback_t callback,
                                               void* callback_args);
};

}  // namespace ucxx
