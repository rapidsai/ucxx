# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD 3-Clause License
# =================================================================================

set(cython_sources arr.pyx libucxx.pyx)
set(linked_libraries ucxx::ucxx ucxx::python Python3::Python)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" ASSOCIATED_TARGETS ucxx
)

find_package(Python REQUIRED COMPONENTS Development NumPy)

set(targets_using_numpy libucxx)
foreach(target IN LISTS targets_using_numpy)
  target_include_directories(${target} PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
  target_compile_definitions(${target} PRIVATE NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
endforeach()
