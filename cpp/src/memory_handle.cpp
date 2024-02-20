/**
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucs/memory/memory_type.h>
#include <ucxx/constructors.h>
#include <ucxx/log.h>
#include <ucxx/memory_handle.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

MemoryHandle::MemoryHandle(std::shared_ptr<Context> context, const size_t size, void* buffer)
{
  setParent(context);

  ucp_mem_map_params_t params;
  if (buffer == nullptr) {
    params = {.field_mask  = UCP_MEM_MAP_PARAM_FIELD_FLAGS | UCP_MEM_MAP_PARAM_FIELD_LENGTH,
              .length      = size,
              .flags       = UCP_MEM_MAP_NONBLOCK | UCP_MEM_MAP_ALLOCATE,
              .memory_type = UCS_MEMORY_TYPE_HOST};
  } else {
    params = {.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH,
              .address     = buffer,
              .length      = size,
              .memory_type = UCS_MEMORY_TYPE_HOST};
  }

  utils::ucsErrorThrow(ucp_mem_map(context->getHandle(), &params, &_handle));

  ucp_mem_attr_t attr = {.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH};

  utils::ucsErrorThrow(ucp_mem_query(_handle, &attr));

  _baseAddress = (uint64_t)attr.address;
  _size        = attr.length;

  ucxx_trace("MemoryHandle created: %p, UCP handle: %p, base address: 0x%lx, size: %lu",
             this,
             _handle,
             _baseAddress,
             _size);
}

MemoryHandle::~MemoryHandle()
{
  ucp_mem_unmap(std::dynamic_pointer_cast<Context>(getParent())->getHandle(), _handle);
  ucxx_trace("ucxx::MemoryHandle destroyed: %p, UCP handle: %p, base address: 0x%lx, size: %lu",
             this,
             _handle,
             _baseAddress,
             _size);
}

std::shared_ptr<MemoryHandle> createMemoryHandle(std::shared_ptr<Context> context,
                                                 const size_t size,
                                                 void* buffer)
{
  return std::shared_ptr<MemoryHandle>(new MemoryHandle(context, size, buffer));
}

ucp_mem_h MemoryHandle::getHandle() { return _handle; }

size_t MemoryHandle::getSize() const { return _size; }

uint64_t MemoryHandle::getBaseAddress() { return _baseAddress; }

std::shared_ptr<RemoteKey> MemoryHandle::createRemoteKey()
{
  return createRemoteKeyFromMemoryHandle(
    std::dynamic_pointer_cast<MemoryHandle>(shared_from_this()));
}

}  // namespace ucxx
