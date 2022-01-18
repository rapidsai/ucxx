/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

namespace ucxx
{

typedef enum
{
    UCXX_REQUEST_STATUS_PENDING = 0,
    UCXX_REQUEST_STATUS_FINISHED = 1,
    UCXX_REQUEST_STATUS_UNITIALIZED = -1,
} ucxx_request_status_t;

typedef struct ucxx_request
{
    bool finished;
    unsigned int uid;
    ucxx_request_status_t status;
} ucxx_request_t;

}  // namespace ucxx
