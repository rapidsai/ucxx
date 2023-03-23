# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

set(ucxx_version 0.0.1)

include(../fetch_rapids.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(ucxx-python)

project(
  ucxx-python
  VERSION ${ucxx_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA
)

option(FIND_UCXX_CPP "Search for existing UCXX C++ installations before defaulting to local files"
       OFF
)
option(UCXX_BUILD_WHEELS "Whether this build is generating a Python wheel." OFF)

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
  if(UCXX_BUILD_WHEELS)
    # Statically link cudart if building wheels
    set(CUDA_STATIC_RUNTIME ON)

    # Don't install the UCXX C++ targets into wheels
    set(_exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_subdirectory(../cpp ucxx-cpp ${_exclude_from_all})

  # Since ucxx._lib requires access to libucxx, we place the library in the ucxx directory
  # and modify the rpaths appropriately.
  set(cython_lib_dir ucxx)
  install(TARGETS ucxx DESTINATION ${cython_lib_dir})
endif()

rapids_cython_init()

add_subdirectory(ucxx/_lib)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET ucxx PATHS "${cython_lib_dir}")
endif()
