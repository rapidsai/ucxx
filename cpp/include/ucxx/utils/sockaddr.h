/**
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

namespace ucxx {

namespace utils {

int sockaddr_set(ucs_sock_addr_t* sockaddr, const char* ip_address, uint16_t port);

void sockaddr_free(ucs_sock_addr_t* sockaddr);

void sockaddr_get_ip_port_str(const struct sockaddr_storage* sock_addr,
                              char* ip_str,
                              char* port_str,
                              size_t max_str_size);

}  // namespace utils

}  // namespace ucxx
