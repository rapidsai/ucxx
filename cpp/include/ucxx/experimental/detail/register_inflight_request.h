/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <tuple>

#include <ucxx/endpoint.h>
#include <ucxx/request.h>
#include <ucxx/worker.h>

namespace ucxx {

class Component;

namespace experimental {

namespace detail {

inline void registerInflightRequest(std::shared_ptr<Component> const& component,
                                    std::shared_ptr<Request> const& req)
{
  if (auto ep = std::dynamic_pointer_cast<Endpoint>(component))
    std::ignore = ep->registerInflightRequest(req);
  else if (auto wk = std::dynamic_pointer_cast<Worker>(component))
    std::ignore = wk->registerInflightRequest(req);
}

}  // namespace detail

}  // namespace experimental

}  // namespace ucxx
