# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD 3-Clause License
# =================================================================================

set(cython_sources python_future_task_app.pyx)
set(linked_libraries ucxx::ucxx ucxx::python Python3::Python)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
)

target_include_directories(python_future_task_app PRIVATE ".")
