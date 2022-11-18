/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <functional>

#include <cuda_runtime_api.h>

#include <ucxx/api.h>

enum class ProgressMode {
  Polling,
  Blocking,
  Wait,
  ThreadPolling,
  ThreadBlocking,
};

void createCudaContextCallback(void* callbackArg);

void waitRequests(std::shared_ptr<ucxx::Worker> worker,
                  std::vector<std::shared_ptr<ucxx::Request>>& requests,
                  std::function<void()>& progressWorker);

void waitRequestsTagMulti(std::shared_ptr<ucxx::Worker> worker,
                          std::vector<std::shared_ptr<ucxx::RequestTagMulti>>& requests,
                          std::function<void()>& progressWorker);

std::function<void()> getProgressFunction(std::shared_ptr<ucxx::Worker> worker,
                                          ProgressMode progressMode);
