# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from posix cimport fcntl

from libc.stdint cimport uint64_t

from libcpp.map cimport map as cpp_map
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string


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
        UCXXContext()
        @staticmethod
        shared_ptr[UCXXContext] create(cpp_map[string, string] ucx_config, uint64_t feature_flags) except +
        shared_ptr[UCXXWorker] createWorker() except +
        cpp_map[string, string] get_config() except +
        ucp_context_h get_handle() except +
        string get_info() except +
