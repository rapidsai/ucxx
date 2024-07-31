# ======================================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 3-Clause License
# ======================================================================================================

rapids_find_package(
  Python3 REQUIRED COMPONENTS Development
)
set(UCXX_PYTHON_LIB Python3::Python)

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------
# Due to https://github.com/openucx/ucx/issues/9614, we cannot export the ucx
# dependency because users would then have no control over whether ucx is found
# multiple times, causing potential configure errors. Therefore, we use a raw
# find_package call instead of rapids_find_package and skip exporting the ucx
# dependency. Consumers of ucxx must find ucx themselves. Once the above issue
# is resolved (see https://github.com/openucx/ucx/pull/9622) we can remove the
# above find_package in favor of the commented out lines below. For the same
# reason, we must also gate this find_package call behind a check for the
# target already existing so that consumers can use tools like CPM.cmake to
# either find or build ucxx from source if it cannot be found (i.e. both cases
# must allow prior finding of ucx).
if(NOT TARGET ucx::ucp)
  find_package(ucx REQUIRED)
endif()
#rapids_find_package(
#  ucx REQUIRED
#  BUILD_EXPORT_SET ucxx-exports
#  INSTALL_EXPORT_SET ucxx-exports
#)

# ##################################################################################################
# * python library --------------------------------------------------------------------------------
add_library(
  ucxx_python
  src/exception.cpp
  src/future.cpp
  src/notifier.cpp
  src/python_future.cpp
  src/python_future_task_collector.cpp
  src/worker.cpp
)

set_target_properties(
  ucxx_python
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             EXPORT_NAME python
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  ucxx_python PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UCXX_CXX_FLAGS}>"
)

get_filename_component(ucxx_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

# Specify include paths for the current target and dependents
target_include_directories(
  ucxx_python
  PUBLIC "$<BUILD_INTERFACE:${ucxx_dir}/include>"
         "$<BUILD_INTERFACE:${ucxx_dir}/python/include>"
  PRIVATE "$<BUILD_INTERFACE:${ucxx_dir}/src>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
)

target_compile_definitions(
  ucxx_python PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${UCXX_CXX_DEFINITIONS}>"
)

target_compile_definitions(ucxx_python PUBLIC UCXX_ENABLE_PYTHON)
# Define spdlog level
target_compile_definitions(ucxx_python PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")

# Specify the target module library dependencies
target_link_libraries(
  ucxx_python
  PUBLIC rmm::rmm ucx::ucp ucxx::ucxx ${UCXX_PYTHON_LIB}
)

# Add Conda library, and include paths if specified
if(TARGET conda_env)
  target_link_libraries(ucxx_python PRIVATE conda_env)
endif()

add_library(ucxx::python ALIAS ucxx_python)
