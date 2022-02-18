/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <chrono>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint.h>
#include <ucxx/typedefs.h>

namespace ucxx
{

class UCXXRequest : public UCXXComponent
{
    private:
        std::shared_ptr<ucxx_request_t> _handle{nullptr};
        std::future<ucs_status_t> _future{};

    UCXXRequest(
            std::shared_ptr<UCXXEndpoint> endpoint,
            std::shared_ptr<ucxx_request_t> request
        ) : _handle{request}, _future{request->completed_promise.get_future()}
    {
        if (endpoint == nullptr || endpoint->getHandle() == nullptr)
            throw ucxx::UCXXError("Endpoint not initialized");

        setParent(endpoint);
    }

    public:

    UCXXRequest() = default;

    UCXXRequest(const UCXXRequest&) = delete;
    UCXXRequest& operator=(UCXXRequest const&) = delete;

    UCXXRequest(UCXXRequest&& o) noexcept
        : _handle{std::exchange(o._handle, nullptr)},
          _future{std::exchange(o._future, std::future<ucs_status_t>{})}
    {
    }

    UCXXRequest& operator=(UCXXRequest&& o) noexcept
    {
        this->_handle = std::exchange(o._handle, nullptr);
        this->_future = std::exchange(o._future, std::future<ucs_status_t>{});

        return *this;
    }

    ~UCXXRequest()
    {
        if (_handle == nullptr)
            return;
    }

    friend std::shared_ptr<UCXXRequest> createRequest(std::shared_ptr<UCXXEndpoint>& endpoint, std::shared_ptr<ucxx_request_t> request)
    {
        return std::shared_ptr<UCXXRequest>(new UCXXRequest(endpoint, request));
    }

    ucs_status_t wait()
    {
        return _future.get();
    }

    template<typename Rep, typename Period>
    bool isCompleted(std::chrono::duration<Rep, Period> period)
    {
        return _future.wait_for(period) == std::future_status::ready;
    }

    bool isCompleted(int64_t periodNs = 0)
    {
        return isCompleted(std::chrono::nanoseconds(periodNs));
    }
};

}  // namespace ucxx
