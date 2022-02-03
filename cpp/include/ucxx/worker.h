/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <ucp/api/ucp.h>

#include <ucxx/constructors.h>
#include <ucxx/component.h>
#include <ucxx/context.h>
#include <ucxx/utils.h>

namespace ucxx
{

class UCXXAddress;
class UCXXEndpoint;
class UCXXListener;

class UCXXWorker : public UCXXComponent
{
    private:
        ucp_worker_h _handle{nullptr};
        int _epoll_fd{-1};
        int _worker_fd{-1};
        int _cancel_efd{-1};

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
        : _handle{std::exchange(o._handle, nullptr)},
          _epoll_fd{std::exchange(o._epoll_fd, -1)},
          _worker_fd{std::exchange(o._worker_fd, -1)},
          _cancel_efd{std::exchange(o._cancel_efd, -1)}
    {
    }

    UCXXWorker& operator=(UCXXWorker&& o) noexcept
    {
        this->_handle = std::exchange(o._handle, nullptr);
        this->_epoll_fd = std::exchange(o._epoll_fd, -1);
        this->_worker_fd = std::exchange(o._worker_fd, -1);
        this->_cancel_efd = std::exchange(o._cancel_efd, -1);

        return *this;
    }

    template <class ...Args>
    friend std::shared_ptr<UCXXWorker> createWorker(Args&& ...args)
    {
        return std::shared_ptr<UCXXWorker>(new UCXXWorker(std::forward<Args>(args)...));
    }


    ~UCXXWorker()
    {
        ucp_worker_destroy(_handle);

        if (_epoll_fd >= 0)
            close(_epoll_fd);
        if (_cancel_efd >= 0)
            close(_cancel_efd);

        _parent->removeChild(this);
    }

    ucp_worker_h get_handle()
    {
        return _handle;
    }

    void init_blocking_progress_mode()
    {
        // In blocking progress mode, we create an epoll file
        // descriptor that we can wait on later.
        // We also introduce an additional eventfd to allow
        // canceling the wait.
        int err;

        assert_ucs_status(ucp_worker_get_efd(_handle, &_worker_fd));

        arm();

        _epoll_fd = epoll_create(1);
        if (_epoll_fd == -1)
            throw std::ios_base::failure("epoll_create(1) returned -1");

        _cancel_efd = eventfd(0, EFD_NONBLOCK);
        if (_cancel_efd < 0)
            throw std::ios_base::failure("eventfd(0, EFD_NONBLOCK) returned -1");


        epoll_event worker_ev = {
            .events = EPOLLIN,
            .data = {
                .fd = _worker_fd,
            }
        };
        epoll_event cancel_ev = {
            .events = EPOLLIN,
            .data = {
                .fd = _cancel_efd,
            }
        };

        err = epoll_ctl(_epoll_fd, EPOLL_CTL_ADD, _worker_fd, &worker_ev);
        if (err != 0)
            throw std::ios_base::failure(std::string("epoll_ctl() returned " + err));
        err = epoll_ctl(_epoll_fd, EPOLL_CTL_ADD, _cancel_efd, &cancel_ev);
        if (err != 0)
            throw std::ios_base::failure(std::string("epoll_ctl() returned " + err));
    }

    bool arm()
    {
        ucs_status_t status = ucp_worker_arm(_handle);
        if (status == UCS_ERR_BUSY)
            return false;
        assert_ucs_status(status);
        return true;
    }

    bool progress_worker_event()
    {
        int ret;
        epoll_event ev;

        if (progress())
            return true;

        if ((_epoll_fd == -1) || !arm())
            return false;

        do {
            ret = epoll_wait(_epoll_fd, &ev, 1, -1);
        } while ((ret == - 1) && (errno == EINTR || errno == EAGAIN));

        return false;
    }

    void cancel_progress_worker_event()
    {
        int err = eventfd_write(_cancel_efd, 1);
        if (err < 0)
            throw std::ios_base::failure(std::string("eventfd_write() returned " + err));
    }

    bool wait_progress()
    {
        assert_ucs_status(ucp_worker_wait(_handle));
        return progress();
    }

    bool progress()
    {
        // Try to progress the communication layer
        return ucp_worker_progress(_handle) != 0;
    }

    std::shared_ptr<UCXXAddress> getAddress()
    {
        auto worker = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
        auto address = ucxx::createAddressFromWorker(worker);
        addChild(std::dynamic_pointer_cast<UCXXComponent>(address));
        return address;
    }

    std::shared_ptr<UCXXEndpoint> createEndpointFromHostname(std::string ip_address, uint16_t port=0, bool endpoint_error_handling=true)
    {
        auto worker = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
        auto endpoint = ucxx::createEndpointFromHostname(worker, ip_address, port, endpoint_error_handling);
        addChild(std::dynamic_pointer_cast<UCXXComponent>(endpoint));
        return endpoint;
    }

    std::shared_ptr<UCXXEndpoint> createEndpointFromWorkerAddress(std::shared_ptr<UCXXAddress> address, bool endpoint_error_handling=true)
    {
        auto worker = std::dynamic_pointer_cast<UCXXWorker>(shared_from_this());
        auto endpoint = ucxx::createEndpointFromWorkerAddress(worker, address, endpoint_error_handling);
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
