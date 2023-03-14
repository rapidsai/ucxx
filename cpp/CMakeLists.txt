# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(../fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(UCX)

project(
    UCXX
  VERSION 22.12.00
  LANGUAGES C CXX CUDA
)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.5)
  message(
    FATAL_ERROR
      "libucxx requires CUDA Toolkit 11.5+ to compile (nvcc ${CMAKE_CUDA_COMPILER_VERSION} provided)"
  )
endif()

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value in first run and subsequent runs.
set(THREADS_PREFER_PTHREAD_FLAG ON)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build benchmarks" OFF)
option(BUILD_EXAMPLES "Configure CMake to build examples" OFF)
option(BUILD_SHARED_LIBS "Build UCXX shared libraries" ON)
option(UCXX_ENABLE_PYTHON "Enable support for Python notifier thread" OFF)
option(UCXX_ENABLE_RMM "Enable support for CUDA multi-buffer transfer with RMM" OFF)
option(DISABLE_DEPRECATION_WARNINGS "Disable warnings generated from deprecated declarations." OFF)
# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
option(CUDA_ENABLE_LINEINFO
       "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF
)
option(CUDA_WARNINGS_AS_ERRORS "Enable -Werror=all-warnings for all CUDA compilation" ON)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

message(VERBOSE "UCXX: Configure CMake to build tests: ${BUILD_TESTS}")
message(VERBOSE "UCXX: Configure CMake to build benchmarks: ${BUILD_BENCHMARKS}")
message(VERBOSE "UCXX: Configure CMake to build examples: ${BUILD_EXAMPLES}")
message(VERBOSE "UCXX: Build UCXX shared libraries: ${BUILD_SHARED_LIBS}")
message(VERBOSE "UCXX: Enable support for Python notifier thread: ${UCXX_ENABLE_PYTHON}")
message(VERBOSE "UCXX: Enable support for CUDA multi-buffer transfer with RMM: ${UCXX_ENABLE_RMM}")
message(
  VERBOSE
  "UCXX: Disable warnings generated from deprecated declarations: ${DISABLE_DEPRECATION_WARNINGS}"
)
message(
  VERBOSE
  "UCXX: Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler: ${CUDA_ENABLE_LINEINFO}"
)
message(VERBOSE "UCXX: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")

# Set a default build type if none was specified
rapids_cmake_build_type("Release")
set(UCXX_BUILD_TESTS ${BUILD_TESTS})
set(UCXX_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})
set(UCXX_BUILD_EXAMPLES ${BUILD_EXAMPLES})

set(UCXX_CXX_FLAGS "")
set(UCXX_CUDA_FLAGS "")
set(UCXX_CXX_DEFINITIONS "")
set(UCXX_CUDA_DEFINITIONS "")

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "UCXX: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

if(NOT UCXX_GENERATED_INCLUDE_DIR)
    set(UCXX_GENERATED_INCLUDE_DIR ${UCXX_BINARY_DIR})
endif()

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET ucxx-exports
  INSTALL_EXPORT_SET ucxx-exports
)
find_package(ucx REQUIRED)
include(cmake/Modules/ConfigureCUDA.cmake) # set other CUDA compilation flags

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------

# find Threads (needed by ucxxtestutil)
rapids_find_package(
  Threads REQUIRED
  BUILD_EXPORT_SET ucxx-exports
  INSTALL_EXPORT_SET ucxx-exports
)

# add third party dependencies using CPM
rapids_cpm_init()
# find rmm
include(cmake/thirdparty/get_rmm.cmake)
# find or install GoogleTest
include(cmake/thirdparty/get_gtest.cmake)

if(UCXX_ENABLE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Development)
  set(UCXX_PYTHON_LIB Python3::Python)
endif()

# ##################################################################################################
# * library targets -------------------------------------------------------------------------------

# Build main library
add_library(
  ucxx
  src/address.cpp
  src/buffer.cpp
  src/component.cpp
  src/config.cpp
  src/context.cpp
  src/delayed_submission.cpp
  src/endpoint.cpp
  src/header.cpp
  src/inflight_requests.cpp
  src/listener.cpp
  src/log.cpp
  src/request.cpp
  src/request_helper.cpp
  src/request_stream.cpp
  src/request_tag.cpp
  src/request_tag_multi.cpp
  src/worker.cpp
  src/worker_progress_thread.cpp
  src/utils/file_descriptor.cpp
  src/utils/sockaddr.cpp
  src/utils/ucx.cpp
)

set_target_properties(
  ucxx
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
  ucxx PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UCXX_CXX_FLAGS}>"
               "$<$<COMPILE_LANGUAGE:CUDA>:${UCXX_CUDA_FLAGS}>"
)

# Specify include paths for the current target and dependents
target_include_directories(
  ucxx
  PUBLIC "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${UCXX_GENERATED_INCLUDE_DIR}/include>"
  PRIVATE "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}/src>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
)

target_compile_definitions(
  ucxx PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${UCXX_CXX_DEFINITIONS}>"
              "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${UCXX_CUDA_DEFINITIONS}>>"
)

# Enable RMM if necessary
if(UCXX_ENABLE_RMM)
    target_compile_definitions(ucxx PUBLIC UCXX_ENABLE_RMM)
endif()

# Define spdlog level
target_compile_definitions(ucxx PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")

# Specify the target module library dependencies
target_link_libraries(
  ucxx
  PUBLIC rmm::rmm ucx::ucp
)

# Add Conda library, and include paths if specified
if(TARGET conda_env)
  target_link_libraries(ucxx PRIVATE conda_env)
endif()

if(CUDA_STATIC_RUNTIME)
  # Tell CMake what CUDA language runtime to use
  set_target_properties(ucxx PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  # Make sure to export to consumers what runtime we used
  target_link_libraries(ucxx PUBLIC CUDA::cudart_static)
else()
  # Tell CMake what CUDA language runtime to use
  set_target_properties(ucxx PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  # Make sure to export to consumers what runtime we used
  target_link_libraries(ucxx PUBLIC CUDA::cudart)
endif()

add_library(ucxx::ucxx ALIAS ucxx)

# Build Python if requested
if(UCXX_ENABLE_PYTHON)
  add_library(
    ucxx_python
    src/python/exception.cpp
    src/python/future.cpp
    src/python/notifier.cpp
    src/python/python_future.cpp
    src/python/worker.cpp
    src/utils/python.cpp
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
           "$<BUILD_INTERFACE:${UCXX_GENERATED_INCLUDE_DIR}/include>"
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

  add_library(ucxx::python ALIAS ucxx_python)
  set(UCXX_PYTHON_TARGET_ALIAS ucxx_python)
else()
  set(UCXX_PYTHON_TARGET_TARGET "")
endif()

# ##################################################################################################
# * tests and benchmarks --------------------------------------------------------------------------
# ##################################################################################################

# ##################################################################################################
# * add tests -------------------------------------------------------------------------------------

if(UCXX_BUILD_TESTS)
  # include CTest module -- automatically calls enable_testing()
  include(CTest)

  # Always print verbose output when tests fail if run using `make test`.
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * add benchmarks --------------------------------------------------------------------------------

if(UCXX_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# * add examples ----------------------------------------------------------------------------------

if(UCXX_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ##################################################################################################
# * install targets -------------------------------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ucxx)

install(
  TARGETS ucxx ${UCXX_PYTHON_TARGET_ALIAS}
  DESTINATION ${lib_dir}
  EXPORT ucxx-exports
)

install(DIRECTORY ${UCXX_SOURCE_DIR}/include/ucxx
                  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

set(doc_string
    [=[
Provide targets for the ucxx library.

UCXX is a C++ interface for the UCX communication framework. It aims to provide
a high-level API for the UCP layer, encompassing both transparent lifetime
management of objects and thread-safety.


Imported Targets
^^^^^^^^^^^^^^^^

If ucxx is found, this module defines the following IMPORTED GLOBAL
targets:

 ucxx::ucxx             - The main ucxx library.
    ]=]
)

rapids_export(
  INSTALL ucxx ${UCXX_PYTHON_TARGET_ALIAS}
  EXPORT_SET ucxx-exports
  GLOBAL_TARGETS ucxx
  NAMESPACE ucxx::
  DOCUMENTATION doc_string
)

# ##################################################################################################
# * build export -------------------------------------------------------------------------------
rapids_export(
  BUILD ucxx
  EXPORT_SET ucxx-exports
  GLOBAL_TARGETS ucxx
  NAMESPACE ucxx::
  DOCUMENTATION doc_string
)

# ##################################################################################################
# * make documentation ----------------------------------------------------------------------------

# doc targets for UCXX
add_custom_command(
  OUTPUT UCXX_DOXYGEN
  WORKING_DIRECTORY ${UCXX_SOURCE_DIR}/doxygen
  COMMAND doxygen Doxyfile
  VERBATIM
  COMMENT "Custom command for building ucxx doxygen docs."
)

add_custom_target(
  docs_ucxx
  DEPENDS UCXX_DOXYGEN
  COMMENT "Custom command for building ucxx doxygen docs."
)

# ##################################################################################################
# * make gdb helper scripts ------------------------------------------------------------------------

# build pretty-printer load script
if(Thrust_SOURCE_DIR AND rmm_SOURCE_DIR)
  configure_file(scripts/load-pretty-printers.in load-pretty-printers @ONLY)
endif()
