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


cdef extern from "<ucxx/python/exception.h>" namespace "ucxx::python" nogil:
    cdef void raise_py_error()


cdef extern from "<ucxx/buffer.h>" namespace "ucxx" nogil:
    # TODO: use `cdef enum class` after moving to Cython 3.x
    ctypedef enum BufferType:
        Host "ucxx::BufferType::Host"
        RMM "ucxx::BufferType::RMM"
        Invalid "ucxx::BufferType::Invalid"

    cdef cppclass Buffer:
        BufferType getType()
        size_t getSize()

    cdef cppclass HostBuffer:
        BufferType getType()
        size_t getSize()
        void* release() except +raise_py_error
        void* data() except +raise_py_error

    cdef cppclass RMMBuffer:
        BufferType getType()
        size_t getSize()
        unique_ptr[device_buffer] release() except +raise_py_error
        void* data() except +raise_py_error


cdef extern from "<ucxx/python/typedefs.h>" namespace "ucxx::python" nogil:
    # TODO: use `cdef enum class` after moving to Cython 3.x
    ctypedef enum RequestNotifierWaitState:
        UcxxPythonRequestNotifierWaitStateReady "ucxx::python::RequestNotifierWaitState::Ready"  # noqa: E501
        UcxxPythonRequestNotifierWaitStateTimeout "ucxx::python::RequestNotifierWaitState::Timeout"  # noqa: E501
        UcxxPythonRequestNotifierWaitStateShutdown "ucxx::python::RequestNotifierWaitState::Shutdown"  # noqa: E501


cdef extern from "<ucxx/api.h>" nogil:
    int UCXX_ENABLE_PYTHON


cdef extern from "<ucxx/api.h>" namespace "ucxx" nogil:
    ctypedef cpp_unordered_map[string, string] ConfigMap

    shared_ptr[Context] createContext(
        ConfigMap ucx_config, uint64_t feature_flags
    ) except +raise_py_error

    cdef cppclass Component:
        shared_ptr[Component] getParent()

    cdef cppclass Context(Component):
        shared_ptr[Worker] createWorker(
            bint enableDelayedSubmission,
            bint enablePythonFuture
        ) except +raise_py_error
        ConfigMap getConfig() except +raise_py_error
        ucp_context_h getHandle()
        string getInfo() except +raise_py_error
        uint64_t getFeatureFlags()

    cdef cppclass Worker(Component):
        ucp_worker_h getHandle()
        shared_ptr[Address] getAddress() except +raise_py_error
        shared_ptr[Endpoint] createEndpointFromHostname(
            string ip_address, uint16_t port, bint endpoint_error_handling
        ) except +raise_py_error
        shared_ptr[Endpoint] createEndpointFromWorkerAddress(
            shared_ptr[Address] address, bint endpoint_error_handling
        ) except +raise_py_error
        shared_ptr[Listener] createListener(
            uint16_t port, ucp_listener_conn_callback_t callback, void *callback_args
        ) except +raise_py_error
        void initBlockingProgressMode() except +raise_py_error
        void progress()
        bint progressOnce()
        void progressWorkerEvent()
        void startProgressThread(bint pollingMode) except +raise_py_error
        void stopProgressThread() except +raise_py_error
        size_t cancelInflightRequests() except +raise_py_error
        bint tagProbe(ucp_tag_t)
        void setProgressThreadStartCallback(
            function[void(void*)] callback, void* callbackArg
        )
        void stopRequestNotifierThread() except +raise_py_error
        RequestNotifierWaitState waitRequestNotifier(
            uint64_t periodNs
        ) except +raise_py_error
        void runRequestNotifier() except +raise_py_error
        void populatePythonFuturesPool() except +raise_py_error

    cdef cppclass Endpoint(Component):
        ucp_ep_h getHandle()
        shared_ptr[Request] streamSend(
            void* buffer, size_t length, bint enable_python_future
        ) except +raise_py_error
        shared_ptr[Request] streamRecv(
            void* buffer, size_t length, bint enable_python_future
        ) except +raise_py_error
        shared_ptr[Request] tagSend(
            void* buffer, size_t length, ucp_tag_t tag, bint enable_python_future
        ) except +raise_py_error
        shared_ptr[Request] tagRecv(
            void* buffer, size_t length, ucp_tag_t tag, bint enable_python_future
        ) except +raise_py_error
        bint isAlive()
        void raiseOnError() except +raise_py_error
        void setCloseCallback(
            function[void(void*)] close_callback, void* close_callback_arg
        )

    cdef cppclass Listener(Component):
        shared_ptr[Endpoint] createEndpointFromConnRequest(
            ucp_conn_request_h conn_request, bint endpoint_error_handling
        ) except +raise_py_error
        uint16_t getPort()

    cdef cppclass Address(Component):
        ucp_address_t* getHandle()
        size_t getLength()
        string getString()

    cdef cppclass Request(Component):
        cpp_bool isCompleted(int64_t period_ns)
        ucs_status_t getStatus()
        void checkError() except +raise_py_error
        PyObject* getPyFuture() except +raise_py_error


cdef extern from "<ucxx/request_tag_multi.h>" namespace "ucxx" nogil:

    ctypedef struct BufferRequest:
        shared_ptr[Request] request
        shared_ptr[string] stringBuffer
        Buffer* buffer

    ctypedef shared_ptr[BufferRequest] BufferRequestPtr

    ctypedef shared_ptr[RequestTagMulti] RequestTagMultiPtr

    cdef cppclass RequestTagMulti:
        vector[BufferRequestPtr] _bufferRequests
        bint _isFilled
        shared_ptr[Endpoint] _endpoint
        ucp_tag_t _tag
        bint _send

        cpp_bool isCompleted(int64_t period_ns)
        ucs_status_t getStatus()
        void checkError() except +raise_py_error
        PyObject* getPyFuture() except +raise_py_error

    RequestTagMultiPtr tagMultiRecv(
        shared_ptr[Endpoint] endpoint,
        ucp_tag_t tag,
        bint enable_python_future,
    ) except +raise_py_error
    RequestTagMultiPtr tagMultiSend(
        shared_ptr[Endpoint] endpoint,
        vector[void*]& buffer,
        vector[size_t]& length,
        vector[int]& isCUDA,
        ucp_tag_t tag,
        bint enable_python_future,
    ) except +raise_py_error

    void tagMultiSendBlocking(
        shared_ptr[Endpoint] endpoint,
        vector[void*]& buffer,
        vector[size_t]& length,
        vector[int]& isCUDA,
        ucp_tag_t tag,
        bint enable_python_future,
    ) except +raise_py_error
    vector[Buffer*] tagMultiRecvBlocking(
        shared_ptr[Endpoint] endpoint,
        ucp_tag_t tag,
        bint enable_python_future,
    ) except +raise_py_error
