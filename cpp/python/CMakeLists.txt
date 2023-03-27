# ======================================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 3-Clause License
# ======================================================================================================

find_package(Python3 REQUIRED COMPONENTS Development)
set(UCXX_PYTHON_LIB Python3::Python)

# ##################################################################################################
# * python library --------------------------------------------------------------------------------
add_library(
  ucxx_python
  src/exception.cpp
  src/future.cpp
  src/notifier.cpp
  src/python_future.cpp
  src/worker.cpp
)

set_target_properties(
  ucxx_python
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  ucxx_python PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UCXX_CXX_FLAGS}>"
                      "$<$<COMPILE_LANGUAGE:CUDA>:${UCXX_CUDA_FLAGS}>"
)

# Specify include paths for the current target and dependents
target_include_directories(
  ucxx_python
  PUBLIC "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}/python/include>"
         "$<BUILD_INTERFACE:${UCXX_GENERATED_INCLUDE_DIR}/include>"
         "$<BUILD_INTERFACE:${UCXX_GENERATED_INCLUDE_DIR}/python/include>"
  PRIVATE "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}/src>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
)

target_compile_definitions(
  ucxx_python PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${UCXX_CXX_DEFINITIONS}>"
                     "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${UCXX_CUDA_DEFINITIONS}>>"
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

if(CUDA_STATIC_RUNTIME)
  # Tell CMake what CUDA language runtime to use
  set_target_properties(ucxx_python PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  # Make sure to export to consumers what runtime we used
  target_link_libraries(ucxx_python PUBLIC CUDA::cudart_static)
else()
  # Tell CMake what CUDA language runtime to use
  set_target_properties(ucxx_python PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  # Make sure to export to consumers what runtime we used
  target_link_libraries(ucxx_python PUBLIC CUDA::cudart)
endif()
