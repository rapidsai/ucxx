# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

# distutils: language = c++
# distutils: extra_compile_args=-std=c++17
# cython: language_level=3

from libc.stdint cimport uintptr_t
from libcpp.memory cimport shared_ptr

from ucxx._lib.libucxx cimport Context, UCXContext
import ucxx._lib.libucxx as ucx_api


cdef test_shared(shared_ptr[Context] ptr):
    assert ptr.get() != NULL
    # assert <uint64_t>ptr.get() == context.ucxx_ptr


cdef get_context(UCXContext context):
    # test_shared(context.get_ucxx_shared_ptr())
    assert <uintptr_t>context.get_ucxx_shared_ptr() == context.ucx_ptr


def test_context_getter():
    # cdef shared_ptr[Context] ucxx_shared_ptr

    feature_flags = [ucx_api.Feature.WAKEUP, ucx_api.Feature.TAG]

    context = ucx_api.UCXContext(feature_flags=tuple(feature_flags))

    get_context(context)

    # test_shared(<UCXContext>context.get_ucxx_shared_ptr())

    # ucxx_shared_ptr = context.get_ucxx_shared_ptr()

    # assert ucxx_shared_ptr.get() != NULL
    # assert <uint64_t>ucxx_shared_ptr.get() == context.ucxx_ptr
