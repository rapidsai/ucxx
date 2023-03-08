/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <memory>
#include <vector>

#include <ucxx/request.h>

namespace ucxx {

void waitSingleRequest(std::shared_ptr<Worker> worker, std::shared_ptr<Request> request)
{
  while (!request->isCompleted())
    worker->progress();
  // while (!request->isCompleted());

  request->checkError();
}

void waitRequests(std::shared_ptr<Worker> worker, std::vector<std::shared_ptr<Request>> requests)
{
  for (auto& r : requests)
    waitSingleRequest(worker, r);
}

}  // namespace ucxx
