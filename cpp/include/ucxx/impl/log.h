/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <algorithm>
#include <cstdlib>

#include <ucxx/log.h>
#include <ucxx/typedefs.h>

namespace ucxx {

// Functions
void parseLogLevel()
{
  std::string logLevelName{};

  if (const char* env = std::getenv("_LOG_LEVEL")) {
    logLevelName = std::string(env);
    std::transform(
      logLevelName.begin(), logLevelName.end(), logLevelName.begin(), [](unsigned char c) {
        return std::toupper(c);
      });

    auto level = logLevelNames.find(logLevelName);
    if (level != logLevelNames.end())
      ucxx_log_component.log_level = (ucs_log_level_t)level->second;
    else
      ucxx_warn("_LOG_LEVEL %s unknown, defaulting to _LOG_LEVEL=%s",
                logLevelName.c_str(),
                logLevelNameDefault.c_str());

    ucxx_info("_LOG_LEVEL: %s", logLevelName.c_str());
  }
}

}  // namespace ucxx
