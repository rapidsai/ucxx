/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <sys/epoll.h>

#include <ucp/api/ucp.h>

#include <ucxx/constructors.h>
#include <ucxx/component.h>
#include <ucxx/context.h>
#include <ucxx/utils.h>

namespace ucxx
{

class UCXXEndpoint;
class UCXXListener;

class UCXXWorker : public UCXXComponent
{
    private:
        ucp_worker_h _handle{nullptr};

    UCXXWorker(std::shared_ptr<ucxx::UCXXContext> context)
    {
        ucp_worker_params_t worker_params;

        if (context == nullptr || context->get_handle() == nullptr)
            throw std::runtime_error("UCXXContext not initialized");

        memset(&worker_params, 0, sizeof(worker_params));
        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
        assert_ucs_status(ucp_worker_create(context->get_handle(), &worker_params, &_handle));

        setParent(std::dynamic_pointer_cast<UCXXComponent>(context));
    }

    public:

    UCXXWorker() = default;

    UCXXWorker(const UCXXWorker&) = delete;
    UCXXWorker& operator=(UCXXWorker const&) = delete;

    UCXXWorker(UCXXWorker&& o) noexcept
        : _handle{std::exchange(o._handle, nullptr)}
    {
    }

    UCXXWorker& operator=(UCXXWorker&& o) noexcept
    {
        this->_handle = std::exchange(o._handle, nullptr);

        return *this;
    }

    template <class ...Args>
    friend std::shared_ptr<UCXXWorker> createWorker(Args&& ...args)
    {
        std::cout << "UCXXWorker::createWorker" << std::endl;
        return std::shared_ptr<UCXXWorker>(new UCXXWorker(std::forward<Args>(args)...));
    }


    ~UCXXWorker()
    {
        ucp_worker_destroy(_handle);

        _parent->removeChild(this);
    }

    ucp_worker_h get_handle()
    {
        return _handle;
    }
    int init_blocking_progress_mode()
    {
        // In blocking progress mode, we create an epoll file
        // descriptor that we can wait on later.
        int ucp_epoll_fd, epoll_fd;
        epoll_event ev;
        int err;

        assert_ucs_status(ucp_worker_get_efd(_handle, &ucp_epoll_fd));
        arm();

        epoll_fd = epoll_create(1);
        if (epoll_fd == -1)
            throw std::ios_base::failure("epoll_create(1) returned -1");
        ev.data.fd = ucp_epoll_fd;
        ev.data.ptr = NULL;
        ev.data.u32 = 0;
        ev.data.u64 = 0;
        ev.events = EPOLLIN;

        err = epoll_ctl(epoll_fd, EPOLL_CTL_ADD, ucp_epoll_fd, &ev);

        if (err != 0)
            throw std::ios_base::failure(std::string("epoll_ctl() returned " + err));

        return epoll_fd;
    }

    bool arm()
    {
        ucs_status_t status = ucp_worker_arm(_handle);
        if (status == UCS_ERR_BUSY)
            return false;
        assert_ucs_status(status);
        return true;
    }

    void progress()
    {
        // Try to progress the communication layer

        // Warning, it is illegal to call this from a call-back function such as
        // the call-back function given to UCXListener, tag_send_nb, and tag_recv_nb.

        while (ucp_worker_progress(_handle) != 0)
        {
        }
    }

    std::shared_ptr<UCXXEndpoint> createEndpointFromHostname(std::string ip_address, uint16_t port=0, bool endpoint_error_handling=true)
    {
        auto worker = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
        auto endpoint = ucxx::createEndpointFromHostname(worker, ip_address, port, endpoint_error_handling);
        addChild(std::dynamic_pointer_cast<UCXXComponent>(endpoint));
        return endpoint;
    }

    std::shared_ptr<UCXXEndpoint> createEndpointFromConnRequest(ucp_conn_request_h conn_request, bool endpoint_error_handling=true)
    {
        auto worker = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
        auto endpoint = ucxx::createEndpointFromConnRequest(worker, conn_request, endpoint_error_handling);
        addChild(std::dynamic_pointer_cast<UCXXComponent>(endpoint));
        return endpoint;
    }

    std::shared_ptr<UCXXListener> createListener(uint16_t port, ucp_listener_conn_callback_t callback, void *callback_args)
    {
        auto worker = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
        auto listener = ucxx::createListener(worker, port, callback, callback_args);
        addChild(std::dynamic_pointer_cast<UCXXComponent>(listener));
        return listener;
    }
};

}  // namespace ucxx
