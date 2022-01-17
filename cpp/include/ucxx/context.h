/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include <ucp/api/ucp.h>

#include <ucxx/utils.h>

namespace ucxx {


class UCXXContext
{
    private:
        ucp_context_h _handle{nullptr};
        std::map<std::string, std::string> _config{};
        uint64_t _feature_flags{0};
        bool _cuda_support{false};

    public:
    static constexpr uint64_t default_feature_flags = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP | UCP_FEATURE_STREAM | UCP_FEATURE_AM | UCP_FEATURE_RMA;

    UCXXContext() = default;

    UCXXContext(const UCXXContext&) = delete;
    UCXXContext& operator=(UCXXContext const&) = delete;

    UCXXContext(UCXXContext&& o) noexcept
        : _handle{std::exchange(o._handle, nullptr)},
          _config{std::exchange(o._config, {})},
          _feature_flags{std::exchange(o._feature_flags, 0)},
          _cuda_support{std::exchange(o._cuda_support, false)}
    {
    }

    UCXXContext& operator=(UCXXContext&& o) noexcept
    {
        this->_handle = std::exchange(o._handle, nullptr);
        this->_config = std::exchange(o._config, {});
        this->_feature_flags = std::exchange(o._feature_flags, 0);
        this->_cuda_support = std::exchange(o._cuda_support, false);

        return *this;
    }

    UCXXContext(std::map<std::string, std::string> ucx_config, uint64_t feature_flags) : _config{ucx_config}, _feature_flags{feature_flags}
    {
        ucp_params_t ucp_params;
        ucs_status_t status;

        // UCP
        std::memset(&ucp_params, 0, sizeof(ucp_params));
        ucp_params.field_mask = (
            UCP_PARAM_FIELD_FEATURES |
            UCP_PARAM_FIELD_REQUEST_SIZE |
            UCP_PARAM_FIELD_REQUEST_INIT
        );
        ucp_params.features = feature_flags;
        ucp_params.request_size = sizeof(ucxx::ucxx_request_t);
        ucp_params.request_init = ucx_py_request_reset;

        ucp_config_t *config = _read_ucx_config(ucx_config);
        status = ucp_init(&ucp_params, config, &this->_handle);

        if (status == UCS_OK)
            this->_config = ucx_config_to_dict(config);

        ucp_config_release(config);

        if (status != UCS_OK)
            throw std::runtime_error("Error calling ucp_init()");

        // UCX supports CUDA if "cuda" is part of the TLS or TLS is "all"
        auto tls = this->_config.find("TLS");
        if (tls != this->_config.end())
            this->_cuda_support = tls->second == "all" || tls->second.find("cuda") != std::string::npos;

        std::cout << "UCP initiated using config: " << std::endl;
        for (const auto& kv : this->_config)
            std::cout << "  " << kv.first << ": " << kv.second << std::endl;
    }

    ~UCXXContext()
    {
        if (this->_handle != nullptr)
            ucp_cleanup(this->_handle);
    }

    std::map<std::string, std::string> get_config()
    {
        return this->_config;
    }

    ucp_context_h get_handle()
    {
        assert(this->initialized);
        return this->_handle;
    }

    std::string get_info()
    {
        assert(this->initialized);

        FILE *text_fd = create_text_fd();
        ucp_context_print_info(this->_handle, text_fd);
        return decode_text_fd(text_fd);
    }
};

} // namespace ucxx
