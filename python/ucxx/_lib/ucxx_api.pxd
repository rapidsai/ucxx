# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

from posix cimport fcntl

cimport numpy as np
from libc.stdint cimport int64_t, uint16_t, uint64_t  # noqa: E402
from libcpp cimport bool as cpp_bool  # noqa: E402
from libcpp.functional cimport function  # noqa: E402
from libcpp.memory cimport shared_ptr, unique_ptr  # noqa: E402
from libcpp.string cimport string  # noqa: E402
from libcpp.unordered_map cimport (  # noqa: E402
    unordered_map as cpp_unordered_map,
)
from libcpp.vector cimport vector  # noqa: E402

# from cython.cimports.cpython.ref import PyObject
# from cpython.ref import PyObject


cdef extern from "Python.h" nogil:
    ctypedef struct PyObject


cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

    enum:
        NPY_ARRAY_OWNDATA


cdef extern from "ucp/api/ucp.h" nogil:
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

    void ucp_get_version(unsigned * major_version,
                         unsigned *minor_version,
                         unsigned *release_number)


cdef extern from "rmm/device_buffer.hpp" namespace "rmm" nogil:
    cdef cppclass device_buffer:
        pass


cdef extern from "<ucxx/python/exception.h>" namespace "ucxx" nogil:
    cdef void raise_py_error()


cdef extern from "<ucxx/buffer_helper.h>" namespace "ucxx" nogil:
    ctypedef void (*UCXXPyBufferDeleter)(void*)

    cdef cppclass UCXXPyBuffer:
        bint isValid()
        size_t getSize()
        bint isCUDA()

    cdef cppclass UCXXPyHostBuffer:
        bint isValid()
        size_t getSize()
        bint isCUDA()
        unique_ptr[void, UCXXPyBufferDeleter] get() except +raise_py_error
        void* release() except +raise_py_error

    ctypedef UCXXPyHostBuffer* UCXXPyHostBufferPtr

    cdef cppclass UCXXPyRMMBuffer:
        bint isValid()
        size_t getSize()
        bint isCUDA()
        unique_ptr[device_buffer] get() except +raise_py_error

    ctypedef UCXXPyRMMBuffer* UCXXPyRMMBufferPtr


cdef extern from "<ucxx/api.h>" nogil:
    int UCXX_ENABLE_PYTHON


cdef extern from "<ucxx/api.h>" namespace "ucxx" nogil:
    ctypedef cpp_unordered_map[string, string] UCXXConfigMap

    shared_ptr[UCXXContext] createContext(
        UCXXConfigMap ucx_config, uint64_t feature_flags
    ) except +raise_py_error

    cdef cppclass UCXXContext:
        shared_ptr[UCXXWorker] createWorker(
            bint enable_delayed_notification
        ) except +raise_py_error
        UCXXConfigMap get_config() except +raise_py_error
        ucp_context_h get_handle()
        string get_info() except +raise_py_error

    cdef cppclass UCXXWorker:
        UCXXWorker(
            shared_ptr[UCXXContext] context, bint enable_delayed_notification
        ) except +
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
        bint progress_once()
        void progress_worker_event()
        void startProgressThread(bint pollingMode) except +raise_py_error
        void stopProgressThread() except +raise_py_error
        size_t cancelInflightRequests() except +raise_py_error
        bint tagProbe(ucp_tag_t)
        void setProgressThreadStartCallback(
            function[void(void*)] callback, void* callback_arg
        )
        void stopRequestNotifierThread() except +raise_py_error
        bint waitRequestNotifier() except +raise_py_error
        void runRequestNotifier() except +raise_py_error
        void populatePythonFuturesPool() except +raise_py_error

    cdef cppclass UCXXEndpoint:
        ucp_ep_h getHandle()
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
        PyObject* getPyFuture() except +raise_py_error


cdef extern from "<ucxx/transfer_tag_multi.h>" namespace "ucxx" nogil:

    ctypedef struct UCXXBufferRequest:
        shared_ptr[UCXXRequest] request
        shared_ptr[string] stringBuffer
        unique_ptr[UCXXPyBuffer] pyBuffer

    ctypedef shared_ptr[UCXXBufferRequest] UCXXBufferRequestPtr

    ctypedef shared_ptr[UCXXBufferRequests] UCXXBufferRequestsPtr

    cdef cppclass UCXXBufferRequests:
        vector[UCXXBufferRequestPtr] _bufferRequests
        bint _isFilled
        shared_ptr[UCXXEndpoint] _endpoint
        ucp_tag_t _tag
        bint _send

    UCXXBufferRequestsPtr tagMultiRecv(
        shared_ptr[UCXXEndpoint] endpoint,
        ucp_tag_t tag,
    ) except +raise_py_error
    UCXXBufferRequestsPtr tagMultiSend(
        shared_ptr[UCXXEndpoint] endpoint,
        vector[void*]& buffer,
        vector[size_t]& length,
        vector[int]& isCUDA,
        ucp_tag_t tag,
    ) except +raise_py_error

    void tagMultiSendBlocking(
        shared_ptr[UCXXEndpoint] endpoint,
        vector[void*]& buffer,
        vector[size_t]& length,
        vector[int]& isCUDA,
        ucp_tag_t tag,
    ) except +raise_py_error
    vector[unique_ptr[UCXXPyBuffer]] tagMultiRecvBlocking(
        shared_ptr[UCXXEndpoint] endpoint,
        ucp_tag_t tag,
    ) except +raise_py_error
