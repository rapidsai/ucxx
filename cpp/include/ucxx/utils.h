/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <cstdio>
#include <exception>
#include <string>
#include <sstream>

#include <ucxx/typedefs.h>

FILE * create_text_fd()
{
    FILE *text_fd = std::tmpfile();
    if (text_fd == nullptr)
        throw std::ios_base::failure("tmpfile() failed");

    return text_fd;
}


std::string decode_text_fd(FILE * text_fd)
{
    size_t size;

    rewind(text_fd);
    fseek(text_fd, 0, SEEK_END);
    size = ftell(text_fd);
    rewind(text_fd);

    std::string text_str(size, '\0');

    if (fread(&text_str[0], sizeof(char), size, text_fd) != size)
        throw std::ios_base::failure("fread() failed");

    fclose(text_fd);

    return text_str;
}


// This function will be called by UCX only on the very first time
// a request memory is initialized
void ucx_py_request_reset(void* request)
{
    ucxx::ucxx_request_t* req = (ucxx::ucxx_request_t*) request;
    req->finished = false;
    req->uid = 0;
    req->status = ucxx::UCXX_REQUEST_STATUS_UNITIALIZED;
}
ucp_config_t * _read_ucx_config(std::map<std::string, std::string> user_options)
{
    ucp_config_t *config;
    ucs_status_t status;
    std::string status_msg;

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK)
    {
        status_msg = ucs_status_string(status);
        throw std::runtime_error(std::string("Couldn't read the UCX options: ") + status_msg);
    }

    // Modify the UCX configuration options based on `config_dict`
    for (const auto& kv : user_options)
    {
        status = ucp_config_modify(config, kv.first.c_str(), kv.second.c_str());
        if (status != UCS_OK)
        {
            ucp_config_release(config);

            if (status == UCS_ERR_NO_ELEM)
            {
                throw std::runtime_error(std::string("Option ") + kv.first + std::string("doesn't exist"));
            }
            else
            {
                throw std::runtime_error(ucs_status_string(status));
            }
        }
    }

    return config;
}


std::map<std::string, std::string> ucx_config_to_dict(ucp_config_t *config)
{
    std::map<std::string, std::string> ret;

    FILE *text_fd = create_text_fd();
    ucp_config_print(config, text_fd, NULL, UCS_CONFIG_PRINT_CONFIG);
    std::istringstream text{decode_text_fd(text_fd)};

    std::string delim = "=";
    std::string line;
    while (std::getline(text, line))
    {
        size_t split = line.find(delim);
        std::string k = line.substr(4, split - 4);  // 4 to strip "UCX_" prefix
        std::string v = line.substr(split + delim.length(), std::string::npos);
        ret[k] = v;
    }

    return ret;
}
