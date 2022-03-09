# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from posix cimport fcntl

from libc.stdint cimport int64_t, uint16_t, uint64_t
from libcpp cimport bool as cpp_bool
from libcpp.functional cimport function
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from .exception import UCXError


cdef extern from "ucp/api/ucp.h":
    # Typedefs
    ctypedef struct ucp_context:
        pass

    ctypedef ucp_context* ucp_context_h

    ctypedef struct ucp_worker:
        pass

    ctypedef ucp_worker* ucp_worker_h

    ctypedef struct ucp_ep:
        pass

    ctypedef ucp_ep* ucp_ep_h

    ctypedef ucp_conn_request* ucp_conn_request_h

    ctypedef struct ucp_conn_request:
        pass

    ctypedef void(*ucp_listener_conn_callback_t)(ucp_conn_request_h request, void *arg)

    ctypedef struct ucp_listener_conn_handler_t:
        ucp_listener_conn_callback_t cb
        void *arg

    ctypedef struct ucp_address_t:
        pass

    ctypedef uint64_t ucp_tag_t

    ctypedef enum ucs_status_t:
        pass

    # Constants
    ucs_status_t UCS_OK

    int UCP_FEATURE_TAG
    int UCP_FEATURE_WAKEUP
    int UCP_FEATURE_STREAM
    int UCP_FEATURE_RMA
    int UCP_FEATURE_AMO32
    int UCP_FEATURE_AMO64
    int UCP_FEATURE_AM

    # Functions
    const char *ucs_status_string(ucs_status_t status)


cdef extern from "<ucxx/exception_py.h>" namespace "ucxx" nogil:
    cdef void raise_py_error()


cdef extern from "<ucxx/api.h>" namespace "ucxx" nogil:
    cdef cppclass UCXXContext:
        UCXXContext()

        @staticmethod
        shared_ptr[UCXXContext] create(
            cpp_map[string, string] ucx_config, uint64_t feature_flags
        ) except +raise_py_error
        shared_ptr[UCXXWorker] createWorker() except +raise_py_error
        cpp_map[string, string] get_config() except +raise_py_error
        ucp_context_h get_handle()
        string get_info() except +raise_py_error

    cdef cppclass UCXXWorker:
        UCXXWorker()
        UCXXWorker(shared_ptr[UCXXContext] context) except +
        ucp_worker_h get_handle()
        shared_ptr[UCXXAddress] getAddress() except +raise_py_error
        shared_ptr[UCXXEndpoint] createEndpointFromHostname(
            string ip_address, uint16_t port, bint endpoint_error_handling
        ) except +raise_py_error
        shared_ptr[UCXXEndpoint] createEndpointFromWorkerAddress(
            shared_ptr[UCXXAddress] address, bint endpoint_error_handling
        ) except +raise_py_error
        shared_ptr[UCXXListener] createListener(
            uint16_t port, ucp_listener_conn_callback_t callback, void *callback_args
        ) except +raise_py_error
        void init_blocking_progress_mode() except +raise_py_error
        void progress()
        void progress_worker_event()
        void startProgressThread() except +raise_py_error
        void stopProgressThread() except +raise_py_error
        size_t cancelInflightRequests() except +raise_py_error
        bint tagProbe(ucp_tag_t)

    cdef cppclass UCXXEndpoint:
        ucp_ep_h* getHandle()
        shared_ptr[UCXXRequest] stream_send(
            void* buffer, size_t length
        ) except +raise_py_error
        shared_ptr[UCXXRequest] stream_recv(
            void* buffer, size_t length
        ) except +raise_py_error
        shared_ptr[UCXXRequest] tag_send(
            void* buffer, size_t length, ucp_tag_t tag
        ) except +raise_py_error
        shared_ptr[UCXXRequest] tag_recv(
            void* buffer, size_t length, ucp_tag_t tag
        ) except +raise_py_error
        bint isAlive()
        void raiseOnError() except +raise_py_error
        void setCloseCallback(
            function[void(void*)] close_callback, void* close_callback_arg
        )

    cdef cppclass UCXXListener:
        shared_ptr[UCXXEndpoint] createEndpointFromConnRequest(
            ucp_conn_request_h conn_request, bint endpoint_error_handling
        ) except +raise_py_error
        uint16_t getPort()

    cdef cppclass UCXXAddress:
        ucp_address_t* getHandle()
        size_t getLength()
        string getString()

    cdef cppclass UCXXRequest:
        cpp_bool isCompleted(int64_t period_ns)
        ucs_status_t getStatus()
        void checkError() except +raise_py_error
