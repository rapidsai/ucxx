/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>
#include <vector>

#include <ucxx/request.h>
#include <ucxx/worker.h>

namespace ucxx {

void waitSingleRequest(std::shared_ptr<Worker> worker, std::shared_ptr<Request> request);

void waitRequests(std::shared_ptr<Worker> worker, std::vector<std::shared_ptr<Request>> requests);

}  // namespace ucxx
