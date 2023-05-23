/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <vector>

#include "include/utils.h"

void createCudaContextCallback(void* callbackArg)
{
  // Force CUDA context creation
  cudaFree(0);
}

void waitRequests(std::shared_ptr<ucxx::Worker> worker,
                  const std::vector<std::shared_ptr<ucxx::Request>>& requests,
                  const std::function<void()>& progressWorker)
{
  for (auto& r : requests) {
    do {
      if (progressWorker) progressWorker();
    } while (!r->isCompleted());
    r->checkError();
  }
}

void waitRequestsTagMulti(std::shared_ptr<ucxx::Worker> worker,
                          const std::vector<std::shared_ptr<ucxx::RequestTagMulti>>& requests,
                          const std::function<void()>& progressWorker)
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
    return std::bind(std::mem_fn(&ucxx::Worker::progressWorkerEvent), worker, -1);
  else if (progressMode == ProgressMode::Wait)
    return std::bind(std::mem_fn(&ucxx::Worker::waitProgress), worker);
  else
    return std::function<void()>();
}
