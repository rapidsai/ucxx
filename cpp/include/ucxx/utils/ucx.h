/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <string>

#include <ucp/api/ucp.h>

namespace ucxx {

namespace utils {

/**
 * @brief Helper function to process UCS return codes.
 *
 * Process a UCS return code to verify if a request has completed successfully (`UCS_OK`)
 * and return `true`, if it is currently in progress (`UCS_INPROGRESS`) and return `false`,
 * or raise an exception if it failed (`UCS_ERR_*`). Additionally set the `msg_context`
 * string to a human-readable error message.
 *
 * @param[in]  status       UCS status for which to check state.
 * @param[out] msg_context  human-readable error message.
 *
 * @throws ucxx::Error  if an error occurred.
 *
 * @returns `true` if the status is UCS_OK to indicate the operation completed inline,
 * and `false` if UCX is still holding user
 */
bool assert_ucs_status(const ucs_status_t status, const std::string& msg_context = "");

}  // namespace utils

}  // namespace ucxx
