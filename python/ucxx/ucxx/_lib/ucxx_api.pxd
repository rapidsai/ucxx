# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


from posix cimport fcntl

from libc.stdint cimport int64_t, uint16_t, uint64_t
from libcpp cimport bool as cpp_bool
from libcpp.functional cimport function
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.optional cimport nullopt_t, optional
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map as cpp_unordered_map
from libcpp.vector cimport vector


cdef extern from "Python.h" nogil:
    ctypedef struct PyObject


cdef extern from "ucp/api/ucp.h" nogil:
    # Typedefs
    ctypedef struct ucp_config_t:
        pass

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

    ctypedef enum ucs_memory_type_t:
        pass

    # Constants
    ucs_status_t UCS_OK

    ucs_memory_type_t UCS_MEMORY_TYPE_HOST
    ucs_memory_type_t UCS_MEMORY_TYPE_CUDA

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
    cdef PyObject* UCXXError

    cdef PyObject* UCXXNoMessageError
    cdef PyObject* UCXXNoResourceError
    cdef PyObject* UCXXIOError
    cdef PyObject* UCXXNoMemoryError
    cdef PyObject* UCXXInvalidParamError
    cdef PyObject* UCXXUnreachableError
    cdef PyObject* UCXXInvalidAddrError
    cdef PyObject* UCXXNotImplementedError
    cdef PyObject* UCXXMessageTruncatedError
    cdef PyObject* UCXXNoProgressError
    cdef PyObject* UCXXBufferTooSmallError
    cdef PyObject* UCXXNoElemError
    cdef PyObject* UCXXSomeConnectsFailedError
    cdef PyObject* UCXXNoDeviceError
    cdef PyObject* UCXXBusyError
    cdef PyObject* UCXXCanceledError
    cdef PyObject* UCXXShmemSegmentError
    cdef PyObject* UCXXAlreadyExistsError
    cdef PyObject* UCXXOutOfRangeError
    cdef PyObject* UCXXTimedOutError
    cdef PyObject* UCXXExceedsLimitError
    cdef PyObject* UCXXUnsupportedError
    cdef PyObject* UCXXRejectedError
    cdef PyObject* UCXXNotConnectedError
    cdef PyObject* UCXXConnectionResetError
    cdef PyObject* UCXXFirstLinkFailureError
    cdef PyObject* UCXXLastLinkFailureError
    cdef PyObject* UCXXFirstEndpointFailureError
    cdef PyObject* UCXXEndpointTimeoutError
    cdef PyObject* UCXXLastEndpointFailureError

    cdef PyObject* UCXXCloseError
    cdef PyObject* UCXXConfigError

    cdef void create_exceptions()
    cdef void raise_py_error()


cdef extern from "<ucxx/python/api.h>" namespace "ucxx::python" nogil:
    shared_ptr[Worker] createPythonWorker "ucxx::python::createWorker"(
        shared_ptr[Context] context,
        bint enableDelayedSubmission,
        bint enableFuture
    ) except +raise_py_error


cdef extern from "<ucxx/buffer.h>" namespace "ucxx" nogil:
    cdef enum class BufferType:
        Host
        RMM
        Invalid

    cdef cppclass Buffer:
        Buffer(const BufferType bufferType, const size_t size_t)
        BufferType getType()
        size_t getSize()

    cdef cppclass HostBuffer:
        HostBuffer(const size_t size_t)
        BufferType getType()
        size_t getSize()
        void* release() except +raise_py_error
        void* data() except +raise_py_error

    cdef cppclass RMMBuffer:
        RMMBuffer(const size_t size_t) except +raise_py_error
        BufferType getType()
        size_t getSize()
        unique_ptr[device_buffer] release() except +raise_py_error
        void* data() except +raise_py_error


cdef extern from "<ucxx/notifier.h>" namespace "ucxx" nogil:
    cdef enum class RequestNotifierWaitState:
        Ready
        Timeout
        Shutdown


cdef extern from "<ucxx/api.h>" namespace "ucxx" nogil:
    cdef enum Tag:
        pass
    cdef enum TagMask:
        pass
    cdef cppclass AmReceiverCallbackInfo:
        pass
    # ctypedef Tag CppTag
    # ctypedef TagMask CppTagMask

    # Using function[Buffer] here doesn't seem possible due to Cython bugs/limitations.
    # The workaround is to use a raw C function pointer and let it be parsed by the
    # compiler.
    # See https://github.com/cython/cython/issues/2041 and
    # https://github.com/cython/cython/issues/3193
    ctypedef shared_ptr[Buffer] (*AmAllocatorType)(size_t)

    ctypedef cpp_unordered_map[string, string] ConfigMap

    shared_ptr[Context] createContext(
        ConfigMap ucx_config, uint64_t feature_flags
    ) except +raise_py_error

    shared_ptr[Address] createAddressFromWorker(shared_ptr[Worker] worker)
    shared_ptr[Address] createAddressFromString(string address_string)

    cdef cppclass Config:
        Config()
        Config(ConfigMap user_options) except +raise_py_error
        ConfigMap get() except +raise_py_error
        ucp_config_t* getHandle()

    cdef cppclass Component:
        shared_ptr[Component] getParent()

    cdef cppclass Context(Component):
        shared_ptr[Worker] createWorker(
            bint enableDelayedSubmission,
            bint enableFuture,
        ) except +raise_py_error
        ConfigMap getConfig() except +raise_py_error
        ucp_context_h getHandle()
        string getInfo() except +raise_py_error
        uint64_t getFeatureFlags()
        bint hasCudaSupport()

    cdef cppclass Worker(Component):
        ucp_worker_h getHandle()
        string getInfo() except +raise_py_error
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
        int getEpollFileDescriptor()
        bint arm() except +raise_py_error
        void progress()
        bint progressOnce()
        void progressWorkerEvent(int epoll_timeout)
        void startProgressThread(
            bint pollingMode, int epoll_timeout
        ) except +raise_py_error
        void stopProgressThread() except +raise_py_error
        size_t cancelInflightRequests(
            uint64_t period, uint64_t maxAttempts
        ) except +raise_py_error
        bint tagProbe(const Tag) const
        void setProgressThreadStartCallback(
            function[void(void*)] callback, void* callbackArg
        )
        void stopRequestNotifierThread() except +raise_py_error
        RequestNotifierWaitState waitRequestNotifier(
            uint64_t periodNs
        ) except +raise_py_error
        void runRequestNotifier() except +raise_py_error
        void populateFuturesPool() except +raise_py_error
        void clearFuturesPool()
        shared_ptr[Request] tagRecv(
            void* buffer,
            size_t length,
            Tag tag,
            TagMask tag_mask,
            bint enable_python_future
        ) except +raise_py_error
        bint isDelayedRequestSubmissionEnabled() const
        bint isFutureEnabled() const
        bint amProbe(ucp_ep_h) const
        void registerAmAllocator(
            ucs_memory_type_t memoryType, AmAllocatorType allocator
        )

    cdef cppclass Endpoint(Component):
        ucp_ep_h getHandle()
        shared_ptr[Request] close(
            bint enable_python_future
        ) except +raise_py_error
        void closeBlocking(uint64_t period, uint64_t maxAttempts)
        shared_ptr[Request] amSend(
            void* buffer,
            size_t length,
            ucs_memory_type_t memory_type,
            # Using `nullopt_t` is a workaround for Cython error
            # "Cannot assign type 'nullopt_t' to 'optional[AmReceiverCallbackInfo]'"
            # Must change when AM receiver callbacks are implemented in Python.
            nullopt_t receiver_callback_info,
            bint enable_python_future
        ) except +raise_py_error
        shared_ptr[Request] amRecv(
            bint enable_python_future
        ) except +raise_py_error
        shared_ptr[Request] streamSend(
            void* buffer, size_t length, bint enable_python_future
        ) except +raise_py_error
        shared_ptr[Request] streamRecv(
            void* buffer, size_t length, bint enable_python_future
        ) except +raise_py_error
        shared_ptr[Request] tagSend(
            void* buffer, size_t length, Tag tag, bint enable_python_future
        ) except +raise_py_error
        shared_ptr[Request] tagRecv(
            void* buffer,
            size_t length,
            Tag tag,
            TagMask tag_mask,
            bint enable_python_future
        ) except +raise_py_error
        shared_ptr[Request] tagMultiSend(
            const vector[void*]& buffer,
            const vector[size_t]& length,
            const vector[int]& isCUDA,
            Tag tag,
            bint enable_python_future
        ) except +raise_py_error
        shared_ptr[Request] tagMultiRecv(
            Tag tag, TagMask tagMask, bint enable_python_future
        ) except +raise_py_error
        bint isAlive()
        void raiseOnError() except +raise_py_error
        void setCloseCallback(
            function[void(ucs_status_t, shared_ptr[void])] close_callback,
            shared_ptr[void] close_callback_arg
        ) except +raise_py_error
        shared_ptr[Worker] getWorker()

    cdef cppclass Listener(Component):
        shared_ptr[Endpoint] createEndpointFromConnRequest(
            ucp_conn_request_h conn_request, bint endpoint_error_handling
        ) except +raise_py_error
        uint16_t getPort()
        string getIp()

    cdef cppclass Address(Component):
        ucp_address_t* getHandle()
        size_t getLength()
        string getString()

    cdef cppclass Request(Component):
        cpp_bool isCompleted()
        ucs_status_t getStatus()
        void checkError() except +raise_py_error
        void* getFuture() except +raise_py_error
        shared_ptr[Buffer] getRecvBuffer() except +raise_py_error
        void cancel()


cdef extern from "<ucxx/request_tag_multi.h>" namespace "ucxx" nogil:

    ctypedef struct BufferRequest:
        shared_ptr[Request] request
        shared_ptr[string] stringBuffer
        shared_ptr[Buffer] buffer

    ctypedef shared_ptr[BufferRequest] BufferRequestPtr

    ctypedef shared_ptr[RequestTagMulti] RequestTagMultiPtr

    cdef cppclass RequestTagMulti:
        vector[BufferRequestPtr] _bufferRequests
        bint _isFilled
        shared_ptr[Endpoint] _endpoint
        bint _send

        cpp_bool isCompleted()
        ucs_status_t getStatus()
        void checkError() except +raise_py_error
        void* getFuture() except +raise_py_error


cdef extern from "<ucxx/utils/python.h>" namespace "ucxx::utils" nogil:
    cpp_bool isPythonAvailable()
