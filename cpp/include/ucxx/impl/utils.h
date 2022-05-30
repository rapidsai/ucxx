/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <cstdio>
#include <exception>
#include <sstream>

#include <ucxx/utils.h>

FILE* create_text_fd()
{
  FILE* text_fd = std::tmpfile();
  if (text_fd == nullptr) throw std::ios_base::failure("tmpfile() failed");

  return text_fd;
}

std::string decode_text_fd(FILE* text_fd)
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

// Helper function to process ucs return codes. Returns True if the status is UCS_OK to
// indicate the operation completed inline, and False if UCX is still holding user
// resources. Raises an error if the return code is an error.
bool assert_ucs_status(const ucs_status_t status, const std::string& msg_context)
{
  std::string msg, ucs_status;

  if (status == UCS_OK) return true;
  if (status == UCS_INPROGRESS) return false;

  // If the status is not OK or INPROGRESS it is an error
  ucs_status = ucs_status_string(status);
  if (!msg_context.empty())
    msg = std::string("[" + msg_context + "] " + std::string(ucs_status));
  else
    msg = ucs_status;
  throw ucxx::Error(msg);
}
