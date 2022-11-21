/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

namespace ucxx {

namespace python {

enum class RequestNotifierWaitState { Ready = 0, Timeout, Shutdown };

}  // namespace python

}  // namespace ucxx
