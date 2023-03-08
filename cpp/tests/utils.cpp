/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <memory>
#include <vector>

#include "utils.h"

void createCudaContextCallback(void* callbackArg)
{
  // Force CUDA context creation
  cudaFree(0);
}

void waitRequests(std::shared_ptr<ucxx::Worker> worker,
                  std::vector<std::shared_ptr<ucxx::Request>>& requests,
                  std::function<void()>& progressWorker)
{
  for (auto& r : requests) {
    do {
      if (progressWorker) progressWorker();
    } while (!r->isCompleted());
    r->checkError();
  }
}

void waitRequestsTagMulti(std::shared_ptr<ucxx::Worker> worker,
                          std::vector<std::shared_ptr<ucxx::RequestTagMulti>>& requests,
                          std::function<void()>& progressWorker)
{
  for (auto& r : requests) {
    do {
      if (progressWorker) progressWorker();
    } while (!r->isCompleted());
    r->checkError();
  }
}

std::function<void()> getProgressFunction(std::shared_ptr<ucxx::Worker> worker,
                                          ProgressMode progressMode)
{
  if (progressMode == ProgressMode::Polling)
    return std::bind(std::mem_fn(&ucxx::Worker::progress), worker);
  else if (progressMode == ProgressMode::Blocking)
    return std::bind(std::mem_fn(&ucxx::Worker::progressWorkerEvent), worker);
  else if (progressMode == ProgressMode::Wait)
    return std::bind(std::mem_fn(&ucxx::Worker::waitProgress), worker);
  else
    return std::function<void()>();
}
