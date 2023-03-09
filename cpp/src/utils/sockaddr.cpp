/**
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#include <ucxx/utils/sockaddr.h>

namespace ucxx {

namespace utils {

int sockaddr_set(ucs_sock_addr_t* sockaddr, const char* ip_address, uint16_t port)
{
  struct sockaddr_in* addr = reinterpret_cast<sockaddr_in*>(malloc(sizeof(struct sockaddr_in)));
  if (addr == NULL) { return 1; }
  memset(addr, 0, sizeof(struct sockaddr_in));
  addr->sin_family      = AF_INET;
  addr->sin_addr.s_addr = ip_address == NULL ? INADDR_ANY : inet_addr(ip_address);
  addr->sin_port        = htons(port);
  sockaddr->addr        = (const struct sockaddr*)addr;
  sockaddr->addrlen     = sizeof(struct sockaddr_in);
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
  struct sockaddr_in addr_in;
  struct sockaddr_in6 addr_in6;

  switch (sockaddr->ss_family) {
    case AF_INET:
      memcpy(&addr_in, sockaddr, sizeof(struct sockaddr_in));
      inet_ntop(AF_INET, &addr_in.sin_addr, ip_str, max_str_size);
      snprintf(port_str, max_str_size, "%d", ntohs(addr_in.sin_port));
    case AF_INET6:
      memcpy(&addr_in6, sockaddr, sizeof(struct sockaddr_in6));
      inet_ntop(AF_INET6, &addr_in6.sin6_addr, ip_str, max_str_size);
      snprintf(port_str, max_str_size, "%d", ntohs(addr_in6.sin6_port));
    default:
      ip_str   = const_cast<char*>(reinterpret_cast<const char*>("Invalid address family"));
      port_str = const_cast<char*>(reinterpret_cast<const char*>("Invalid address family"));
  }
}

}  // namespace utils

}  // namespace ucxx
