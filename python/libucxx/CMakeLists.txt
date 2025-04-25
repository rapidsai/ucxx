# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

file(READ "${CMAKE_CURRENT_LIST_DIR}/../../VERSION" _version_contents)
if(_version_contents MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+).*$")
  set(ucxx_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
else()
  string(REPLACE "\n" "\n  " _version_contents_formatted "${_version_contents}")
  message(
    FATAL_ERROR
      "Could not determine ucxx version. Contents of VERSION file:\n  ${_version_contents_formatted}"
  )
endif()

include(../../cmake/rapids_config.cmake)
include(rapids-cpm)
include(rapids-cuda)
rapids_cuda_init_architectures(libucxx-python)
rapids_cpm_init()

project(
  libucxx-python
  VERSION "${ucxx_version}"
  LANGUAGES CXX
)

# Check if ucxx is already available. If so, it's the user's responsibility to ensure that the CMake
# package is also available at build time of the Python ucxx package.
find_package(ucxx "${ucxx_version}")

if(ucxx_FOUND)
  return()
endif()

unset(ucxx_FOUND)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)

add_subdirectory(../../cpp ucxx-cpp)
