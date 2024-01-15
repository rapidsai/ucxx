# ======================================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 3-Clause License
# ======================================================================================================

rapids_find_package(
  Python3 REQUIRED COMPONENTS Development
  BUILD_EXPORT_SET ucxx-python-exports
  INSTALL_EXPORT_SET ucxx-python-exports
)
set(UCXX_PYTHON_LIB Python3::Python)

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

# Specify include paths for the current target and dependents
target_include_directories(
  ucxx_python
  PUBLIC "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}/python/include>"
  PRIVATE "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}/src>"
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

include(GNUInstallDirs)
add_library(ucxx::python ALIAS ucxx_python)
install(DIRECTORY ${UCXX_SOURCE_DIR}/python/include/ucxx
                  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
                  COMPONENT python
                  EXCLUDE_FROM_ALL
)

include("${rapids-cmake-dir}/export/find_package_root.cmake")
rapids_export_find_package_root(
  BUILD Python3 [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET ucxx-python-exports
)
