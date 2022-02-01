/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/sockaddr_utils.h>
#include <ucxx/utils.h>
#include <ucxx/worker.h>

namespace ucxx
{

void _ucp_listener_destroy(ucp_listener_h ptr)
{
    if (ptr != nullptr)
        ucp_listener_destroy(ptr);
}

class UCXXListener : public UCXXComponent
{
    private:
        std::unique_ptr<ucp_listener, void(*)(ucp_listener_h)> _handle{nullptr, _ucp_listener_destroy};
        std::string _ip{};
        uint16_t _port{0};

    UCXXListener(
            std::shared_ptr<UCXXWorker> worker,
            uint16_t port,
            ucp_listener_conn_callback_t callback,
            void *callback_args
        )
    {
        if (worker == nullptr || worker->get_handle() == nullptr)
            throw ucxx::UCXXError("Worker not initialized");

        ucp_listener_params_t params;
        ucp_listener_attr_t attr;

        params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                            UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
        params.conn_handler.cb = callback;
        params.conn_handler.arg = callback_args;

        if (sockaddr_utils_set(&params.sockaddr, NULL, port))
            // throw std::bad_alloc("Failed allocation of sockaddr")
            throw std::bad_alloc();
        std::unique_ptr<ucs_sock_addr_t, void(*)(ucs_sock_addr_t*)> sockaddr(&params.sockaddr, sockaddr_utils_free);

        ucp_listener_h handle = nullptr;
        assert_ucs_status(ucp_listener_create(worker->get_handle(), &params, &handle));
        _handle = std::unique_ptr<ucp_listener, void(*)(ucp_listener_h)>(handle, ucp_listener_destroy);

        attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
        assert_ucs_status(ucp_listener_query(_handle.get(), &attr));

        size_t MAX_STR_LEN = 50;
        char ip_str[MAX_STR_LEN];
        char port_str[MAX_STR_LEN];
        sockaddr_utils_get_ip_port_str(&attr.sockaddr,
                                       ip_str,
                                       port_str,
                                       MAX_STR_LEN);

        _ip = std::string(ip_str);
        _port = (uint16_t)atoi(port_str);

        setParent(worker);
    }

    public:

    UCXXListener() = default;

    UCXXListener(const UCXXListener&) = delete;
    UCXXListener& operator=(UCXXListener const&) = delete;

    UCXXListener(UCXXListener&& o) noexcept
        : _handle{std::exchange(o._handle, nullptr)}
    {
    }

    UCXXListener& operator=(UCXXListener&& o) noexcept
    {
        this->_handle = std::exchange(o._handle, nullptr);

        return *this;
    }

    ~UCXXListener()
    {
        _parent->removeChild(this);
    }

    friend std::shared_ptr<UCXXListener> createListener(
            std::shared_ptr<UCXXWorker> worker,
            uint16_t port,
            ucp_listener_conn_callback_t callback,
            void *callback_args
        )
    {
        return std::shared_ptr<UCXXListener>(new UCXXListener(worker, port, callback, callback_args));
    }

    ucp_listener_h get_handle()
    {
        return _handle.get();
    }
};

}  // namespace ucxx
