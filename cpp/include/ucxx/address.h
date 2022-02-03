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
#include <ucxx/worker.h>

namespace ucxx
{

class UCXXAddress : public UCXXComponent
{
    private:
        char* _handle{nullptr};
        size_t _length{0};

        UCXXAddress(ucp_address_t* address, size_t length, std::shared_ptr<ucxx::UCXXWorker> worker) : _handle{(char*)address}, _length{length}
        {
            if (worker != nullptr)
                setParent(worker);
        }

    public:

        UCXXAddress() = default;

        UCXXAddress(const UCXXAddress&) = delete;
        UCXXAddress& operator=(UCXXAddress const&) = delete;

        UCXXAddress(UCXXAddress&& o) noexcept
            : _handle{std::exchange(o._handle, nullptr)},
              _length{std::exchange(o._length, 0)}
        {
        }

        UCXXAddress& operator=(UCXXAddress&& o) noexcept
        {
            this->_handle = std::exchange(o._handle, nullptr);
            this->_length = std::exchange(o._length, 0);

            return *this;
        }

        ~UCXXAddress()
        {
            if (_handle == nullptr)
                return;

            auto worker = std::dynamic_pointer_cast<UCXXWorker>(getParent());
            if (worker == nullptr)
            {
                delete[] _handle;
            }
            else
            {
                ucp_worker_release_address(worker->get_handle(), (ucp_address_t*)_handle);
            }
        }

        friend std::shared_ptr<UCXXAddress> createAddressFromWorker(std::shared_ptr<ucxx::UCXXWorker> worker)
        {
            ucp_worker_h ucp_worker = worker->get_handle();
            ucp_address_t* address{nullptr};
            size_t length = 0;

            assert_ucs_status(ucp_worker_get_address(ucp_worker, &address, &length));
            return std::shared_ptr<UCXXAddress>(new UCXXAddress(address, length, worker));
        }

        friend std::shared_ptr<UCXXAddress> createAddressFromString(std::string addressString)
        {
            char* address = new char[addressString.length()];
            size_t length = addressString.length();
            memcpy((char*)address, addressString.c_str(), length);
            return std::shared_ptr<UCXXAddress>(new UCXXAddress((ucp_address_t*)address, length, nullptr));
        }

        ucp_address_t* getHandle() const
        {
            return (ucp_address_t*)_handle;
        }

        size_t getLength() const
        {
            return _length;
        }

        std::string getString() const
        {
            return std::string{(char*)_handle, _length};
        }
};

}  // namespace ucxx
