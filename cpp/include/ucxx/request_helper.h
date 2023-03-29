/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
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
