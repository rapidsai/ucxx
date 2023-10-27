# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD 3-Clause License
# =================================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

set(ucxx_version 0.35.00)

include(../fetch_rapids.cmake)

project(
  ucxx-python
  VERSION ${ucxx_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX
)

option(FIND_UCXX_CPP "Search for existing UCXX C++ installations before defaulting to local files"
       OFF
)

# If the user requested it we attempt to find UCXX.
if(FIND_UCXX_CPP)
  find_package(ucxx ${ucxx_version} REQUIRED COMPONENTS python)
else()
  set(ucxx_FOUND OFF)
endif()

include(rapids-cython)

if(NOT ucxx_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)

  set(_exclude_from_all "")

  add_subdirectory(../cpp ucxx-cpp ${_exclude_from_all})

  # Since ucxx._lib requires access to libucxx, we place the library in the ucxx directory
  # and modify the rpaths appropriately.
  set(cython_lib_dir ucxx)
  install(TARGETS ucxx DESTINATION ${cython_lib_dir})
endif()

rapids_cython_init()

add_subdirectory(ucxx/examples)
add_subdirectory(ucxx/_lib)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET ucxx PATHS "${cython_lib_dir}")
  rapids_cython_add_rpath_entries(TARGET ucxx_examples PATHS "${cython_lib_dir}")
endif()
