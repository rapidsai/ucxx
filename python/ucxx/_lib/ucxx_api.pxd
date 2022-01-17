# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from posix cimport fcntl

from libc.stdint cimport uint64_t

from libcpp.string cimport string
from libcpp.map cimport map


cdef extern from "ucp/api/ucp.h":
    ctypedef struct ucp_context:
        pass

    ctypedef ucp_context* ucp_context_h

    int UCP_FEATURE_TAG
    int UCP_FEATURE_WAKEUP
    int UCP_FEATURE_STREAM
    int UCP_FEATURE_RMA
    int UCP_FEATURE_AMO32
    int UCP_FEATURE_AMO64
    int UCP_FEATURE_AM


cdef extern from "<ucxx/context.h>" namespace "ucxx" nogil:
    cdef cppclass UCXXContext:
        UCXXContext() except +
        UCXXContext(map[string, string] ucx_config, uint64_t feature_flags) except +
        map[string, string] get_config()
        ucp_context_h get_handle()
        string get_info()
