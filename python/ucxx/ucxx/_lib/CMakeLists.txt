# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
# cmake-format: on
# =================================================================================

set(cython_sources arr.pyx libucxx.pyx)
set(linked_libraries ucxx::ucxx ucxx::python)

if(UCXX_ENABLE_CCCL)
  string(APPEND CYTHON_FLAGS " -E UCXX_ENABLE_CCCL=True")
endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" ASSOCIATED_TARGETS ucxx ucxx_python COMPONENT ucxx
)
