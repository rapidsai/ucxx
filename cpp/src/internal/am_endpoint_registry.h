/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <memory>

#include <ucp/api/ucp.h>

namespace ucxx {

class Endpoint;
class RequestAm;
class Worker;

namespace internal {

/**
 * @brief Internal access point for active-message endpoint receive state.
 *
 * This class centralizes endpoint lifecycle and receive-queue access to Worker-owned AM state.
 */
class AmEndpointRegistry {
 public:
  [[nodiscard]] static std::shared_ptr<RequestAm> getAmRecv(
    Worker* worker,
    Endpoint* endpoint,
    std::function<std::shared_ptr<RequestAm>()> createAmRecvRequestFunction);
  static void createEndpoint(Worker* worker,
                             Endpoint* endpoint,
                             std::function<ucp_ep_h()> createEndpointFunction);
  static void markEndpointClosed(Worker* worker, Endpoint* endpoint);
  static void closeEndpoint(Worker* worker, ucp_ep_h ep, Endpoint* endpoint);
  static void releaseEndpoint(Worker* worker, Endpoint* endpoint);
  [[nodiscard]] static bool probe(const Worker* worker, ucp_ep_h endpointHandle);
  [[nodiscard]] static bool probeEndpoint(const Worker* worker, const Endpoint* endpoint);
};

}  // namespace internal
}  // namespace ucxx
