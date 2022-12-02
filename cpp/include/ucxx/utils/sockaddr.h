/**
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#pragma once

#include <ucp/api/ucp.h>

namespace ucxx {

namespace utils {

/**
 * @brief Set socket address and port of a socket address storage.
 *
 * Set a socket address and port as defined by the user in a socket address storage that
 * may later be used to specify an address to bind a UCP listener to.
 *
 * @param[in] sockaddr    pointer to the UCS socket address storage.
 * @param[in] ip_address  valid socket address (e.g., IP address or hostname) or NULL as a
 *                        wildcard for "all" to set the socket address storage to.
 * @param[in] port        port to set the socket address storaget to.
 */
int sockaddr_set(ucs_sock_addr_t* sockaddr, const char* ip_address, uint16_t port);

/**
 * @brief Release the underlying socket address.
 *
 * Release the underlying socket address container.
 *
 * NOTE: This function does not release the `ucs_sock_addr_t`, only the underlying
 * `sockaddr` member.
 *
 * @param[in] sockaddr  pointer to the UCS socket address storage.
 */
void sockaddr_free(ucs_sock_addr_t* sockaddr);

/**
 * @brief Get socket address and port of a socket address storage.
 *
 * Get the socket address (usually the IP address) and port from a socket address storage
 * pointer.
 *
 * @param[in] sock_addr     pointer to the socket address storage.
 * @param[in] ip_str        socket address (or IP) contained the socket address storage.
 * @param[in] port_str      port contained the socket address storage.
 * @param[in] max_str_size  size of the `ip_str` and `port_str` strings.
 */
void sockaddr_get_ip_port_str(const struct sockaddr_storage* sock_addr,
                              char* ip_str,
                              char* port_str,
                              size_t max_str_size);

}  // namespace utils

}  // namespace ucxx
