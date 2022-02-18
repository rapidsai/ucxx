# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from posix cimport fcntl

from libc.stdint cimport int64_t, uint16_t, uint64_t

from libcpp cimport bool as cpp_bool
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport shared_ptr
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from .exception import UCXError


cdef extern from "<future>" namespace "std" nogil:
    cdef cppclass future[T]:
        # future() except +
        T get() except +

    cdef cppclass promise[T]:
        future[T] get_future() except +


cdef extern from "ucp/api/ucp.h":
    # Typedefs
    ctypedef struct ucp_context:
        pass

    ctypedef ucp_context* ucp_context_h

    ctypedef struct ucp_worker:
        pass

    ctypedef ucp_worker* ucp_worker_h

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


cdef extern from "<ucxx/typedefs.h>" namespace "ucxx" nogil:
    ctypedef struct ucxx_request_t:
        promise[ucs_status_t] completed_promise;


cdef extern from "<ucxx/context.h>" namespace "ucxx" nogil:
    cdef cppclass UCXXContext:
        UCXXContext()
        @staticmethod
        shared_ptr[UCXXContext] create(cpp_map[string, string] ucx_config, uint64_t feature_flags) except +
        shared_ptr[UCXXWorker] createWorker() except +
        cpp_map[string, string] get_config() except +
        ucp_context_h get_handle() except +
        string get_info() except +


cdef extern from "<ucxx/worker.h>" namespace "ucxx" nogil:
    cdef cppclass UCXXWorker:
        UCXXWorker()
        UCXXWorker(shared_ptr[UCXXContext] context) except +
        ucp_worker_h get_handle() except +
        shared_ptr[UCXXAddress] getAddress() except +
        shared_ptr[UCXXEndpoint] createEndpointFromHostname(string ip_address, uint16_t port, bint endpoint_error_handling) except +
        shared_ptr[UCXXEndpoint] createEndpointFromWorkerAddress(shared_ptr[UCXXAddress] address, bint endpoint_error_handling) except +
        shared_ptr[UCXXListener] createListener(uint16_t port, ucp_listener_conn_callback_t callback, void *callback_args) except +
        void init_blocking_progress_mode() except +
        void progress() except +
        void progress_worker_event() except +
        void startProgressThread() except +
        void stopProgressThread() except +


cdef extern from "<ucxx/endpoint.h>" namespace "ucxx" nogil:
    cdef cppclass UCXXEndpoint:
        shared_ptr[UCXXRequest] tag_send(void* buffer, size_t length, ucp_tag_t tag) except +
        shared_ptr[UCXXRequest] tag_recv(void* buffer, size_t length, ucp_tag_t tag) except +


cdef extern from "<ucxx/listener.h>" namespace "ucxx" nogil:
    cdef cppclass UCXXListener:
        shared_ptr[UCXXEndpoint] createEndpointFromConnRequest(ucp_conn_request_h conn_request, bint endpoint_error_handling) except +


cdef extern from "<ucxx/address.h>" namespace "ucxx" nogil:
    cdef cppclass UCXXAddress:
        ucp_address_t* getHandle()
        size_t getLength()
        string getString()


cdef extern from "<ucxx/request.h>" namespace "ucxx" nogil:
    cdef cppclass UCXXRequest:
        ucs_status_t wait()
        cpp_bool isCompleted(int64_t period_ns)
