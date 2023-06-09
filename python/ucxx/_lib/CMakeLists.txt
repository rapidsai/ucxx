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

# TODO: Finding NumPy currently requires finding Development due to a bug in CMake. This bug was
# fixed in https://gitlab.kitware.com/cmake/cmake/-/merge_requests/7410 and will be available in
# CMake 3.24, so we can remove the Development component once we upgrade to CMake 3.24.
# find_package(Python REQUIRED COMPONENTS Development NumPy)

# Note: The bug noted above prevents us from finding NumPy successfully using FindPython.cmake
# inside the manylinux images used to build wheels because manylinux images do not contain
# libpython.so and therefore Development cannot be found. Until we upgrade to CMake 3.24, we should
# use FindNumpy.cmake instead (provided by scikit-build). When we switch to 3.24 we can try
# switching back, but it may not work if that implicitly still requires Python libraries. In that
# case we'll need to follow up with the CMake team to remove that dependency.  The stopgap solution
# is to unpack the static lib tarballs in the wheel building jobs so that there are at least static
# libs to be found, but that should be a last resort since it implies a dependency that isn't really
# necessary. The relevant command is tar -xf /opt/_internal/static-libs-for-embedding-only.tar.xz -C
# /opt/_internal"
find_package(NumPy REQUIRED)
set(targets_using_numpy libucxx)
foreach(target IN LISTS targets_using_numpy)
  target_include_directories(${target} PRIVATE "${NumPy_INCLUDE_DIRS}")
  # Switch to the line below when we switch back to FindPython.cmake in CMake 3.24.
  # target_include_directories(${target} PRIVATE "${Python_NumPy_INCLUDE_DIRS}")

  target_compile_definitions(${target} PRIVATE NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
endforeach()
