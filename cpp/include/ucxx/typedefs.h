/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <future>

namespace ucxx
{

typedef struct ucxx_request
{
    std::promise<ucs_status_t> completed_promise;
} ucxx_request_t;

}  // namespace ucxx
