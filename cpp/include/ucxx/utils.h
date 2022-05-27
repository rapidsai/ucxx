/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <cstdio>
#include <string>

#include <ucp/api/ucp.h>

FILE* create_text_fd();

std::string decode_text_fd(FILE* text_fd);

// Helper function to process ucs return codes. Returns True if the status is UCS_OK to
// indicate the operation completed inline, and False if UCX is still holding user
// resources. Raises an error if the return code is an error.
bool assert_ucs_status(const ucs_status_t status, const std::string& msg_context = "");
