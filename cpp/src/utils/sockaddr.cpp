/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <arpa/inet.h>
#include <memory>
#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#include <ucxx/exception.h>
#include <ucxx/utils/sockaddr.h>

namespace ucxx {

namespace utils {

int sockaddr_set(ucs_sock_addr_t* sockaddr, const char* ip_address, uint16_t port)
{
  std::unique_ptr<struct addrinfo, void (*)(struct addrinfo*)> info(nullptr, ::freeaddrinfo);
  {
    char ports[6];
    struct addrinfo* result = nullptr;
    struct addrinfo hints;
    // Don't restrict lookups
    ::memset(&hints, 0, sizeof(hints));
    // Except, port is numeric, address may be NULL meaning the
    // returned address is the wildcard.
    hints.ai_flags = AI_NUMERICSERV | AI_PASSIVE;
    if (::snprintf(ports, sizeof(ports), "%u", port) > sizeof(ports))
      throw ucxx::Error(std::string("Invalid port"));
    if (::getaddrinfo(ip_address, ports, &hints, &result))
      throw ucxx::Error(std::string("Invalid IP address or hostname"));
    info.reset(result);
  }
  void* x = ::malloc(info->ai_addrlen);
  ::memcpy(x, info->ai_addr, info->ai_addrlen);
  sockaddr->addr    = reinterpret_cast<struct sockaddr*>(x);
  sockaddr->addrlen = info->ai_addrlen;
  return 0;
}

void sockaddr_free(ucs_sock_addr_t* sockaddr)
{
  ::free(const_cast<void*>(reinterpret_cast<const void*>(sockaddr->addr)));
}

void sockaddr_get_ip_port_str(const struct sockaddr_storage* sockaddr,
                              char* ip_str,
                              char* port_str,
                              size_t max_str_size)
{
  const struct sockaddr_in* addr_in   = nullptr;
  const struct sockaddr_in6* addr_in6 = nullptr;

  switch (sockaddr->ss_family) {
    case AF_INET:
      addr_in = reinterpret_cast<decltype(addr_in)>(sockaddr);
      inet_ntop(AF_INET, &addr_in->sin_addr, ip_str, max_str_size);
      snprintf(port_str, max_str_size, "%u", ntohs(addr_in->sin_port));
    case AF_INET6:
      addr_in6 = reinterpret_cast<decltype(addr_in6)>(sockaddr);
      inet_ntop(AF_INET6, &addr_in6->sin6_addr, ip_str, max_str_size);
      snprintf(port_str, max_str_size, "%u", ntohs(addr_in6->sin6_port));
    default:
      ip_str   = const_cast<char*>(reinterpret_cast<const char*>("Invalid address family"));
      port_str = const_cast<char*>(reinterpret_cast<const char*>("Invalid address family"));
  }
}

}  // namespace utils

}  // namespace ucxx
