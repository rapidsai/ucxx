# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD 3-Clause License
# =================================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

file(READ "${CMAKE_CURRENT_LIST_DIR}/../VERSION" _version_contents)
if(_version_contents MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+).*$")
  set(ucxx_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
else()
  string(REPLACE "\n" "\n  " _version_contents_formatted "${_version_contents}")
  message(FATAL_ERROR "Could not determine ucxx version. Contents of VERSION file:\n  ${_version_contents_formatted}")
endif()

include(../../fetch_rapids.cmake)

project(
  ucxx-python
  VERSION ${ucxx_version}
  LANGUAGES CXX
)

option(FIND_UCXX_CPP "Search for existing UCXX C++ installations before defaulting to local files"
       OFF
)

# If the user requested it we attempt to find UCXX.
if(FIND_UCXX_CPP)
  find_package(ucx REQUIRED)
  find_package(ucxx ${ucxx_version} REQUIRED COMPONENTS python)
else()
  set(ucxx_FOUND OFF)
endif()

include(rapids-cython-core)

if(NOT ucxx_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  set(UCXX_ENABLE_PYTHON ON)

  set(_exclude_from_all "")

  add_subdirectory(../../cpp ucxx-cpp ${_exclude_from_all})

  # Since ucxx._lib requires access to libucxx, we place the library in the ucxx directory
  # and modify the rpaths appropriately.
  set(cython_lib_dir ucxx)
  install(TARGETS ucxx DESTINATION ${cython_lib_dir})
  install(TARGETS ucxx_python DESTINATION ${cython_lib_dir})
endif()

rapids_cython_init()

find_package(
  Python3 REQUIRED COMPONENTS Development.Embed
)
add_subdirectory(ucxx/examples)
add_subdirectory(ucxx/_lib)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET ucxx PATHS "${cython_lib_dir}")
endif()
