# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


import asyncio
import enum
import functools
import logging
import warnings
import weakref

from cpython.buffer cimport PyBUF_FORMAT, PyBUF_ND, PyBUF_WRITABLE
from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref
from libc.stdint cimport uint8_t, uintptr_t
from libc.stdlib cimport free
from libcpp cimport nullptr
from libcpp.functional cimport function
from libcpp.memory cimport (
    dynamic_pointer_cast,
    make_shared,
    make_unique,
    shared_ptr,
    static_pointer_cast,
    unique_ptr,
)
from libcpp.optional cimport nullopt
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

import numpy as np

from rmm.pylibrmm.device_buffer cimport DeviceBuffer

from .arr cimport Array
from .ucxx_api cimport *

include "tag.pyx"

logger = logging.getLogger("ucx")


cdef class HostBufferAdapter:
    """A simple adapter around HostBuffer implementing the buffer protocol"""
    cdef Py_ssize_t _size
    cdef void* _ptr
    cdef Py_ssize_t[1] _shape
    cdef Py_ssize_t[1] _strides
    cdef Py_ssize_t _itemsize

    @staticmethod
    cdef _from_host_buffer(HostBuffer* host_buffer):
        """Construct a new HostBufferAdapter from a HostBuffer.

        This factory takes ownership of the input host_buffer's data, so
        attempting to use the input after this function is called will result
        in undefined behavior.
        """
        cdef HostBufferAdapter obj = HostBufferAdapter.__new__(HostBufferAdapter)
        obj._size = host_buffer.getSize()
        obj._ptr = host_buffer.release()
        obj._shape = [obj._size]
        obj._itemsize = sizeof(uint8_t)
        obj._strides = [obj._itemsize]
        return obj

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.buf = self._ptr
        buffer.format = 'B'
        buffer.internal = NULL
        buffer.itemsize = self._itemsize
        buffer.len = self._size * self._itemsize
        buffer.ndim = 1
        buffer.readonly = 0
        buffer.shape = self._shape
        buffer.strides = self._strides
        buffer.suboffsets = NULL
        buffer.obj = self

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __dealloc__(self):
        free(self._ptr)


def _get_rmm_buffer(uintptr_t recv_buffer_ptr):
    cdef RMMBuffer* rmm_buffer = <RMMBuffer*>recv_buffer_ptr
    return DeviceBuffer.c_from_unique_ptr(move(rmm_buffer.release()))


def _get_host_buffer(uintptr_t recv_buffer_ptr):
    cdef HostBuffer* host_buffer = <HostBuffer*>recv_buffer_ptr
    return np.asarray(HostBufferAdapter._from_host_buffer(host_buffer))


cdef shared_ptr[Buffer] _rmm_am_allocator(size_t length) noexcept nogil:
    cdef shared_ptr[RMMBuffer] rmm_buffer = make_shared[RMMBuffer](length)
    return dynamic_pointer_cast[Buffer, RMMBuffer](rmm_buffer)


###############################################################################
#                               Exceptions                                    #
###############################################################################

UCXError = None

UCXNoMessageError = None
UCXNoResourceError = None
UCXIOError = None
UCXNoMemoryError = None
UCXInvalidParamError = None
UCXUnreachableError = None
UCXInvalidAddrError = None
UCXNotImplementedError = None
UCXMessageTruncatedError = None
UCXNoProgressError = None
UCXBufferTooSmallError = None
UCXNoElemError = None
UCXSomeConnectsFailedError = None
UCXNoDeviceError = None
UCXBusyError = None
UCXCanceledError = None
UCXShmemSegmentError = None
UCXAlreadyExistsError = None
UCXOutOfRangeError = None
UCXTimedOutError = None
UCXExceedsLimitError = None
UCXUnsupportedError = None
UCXRejectedError = None
UCXNotConnectedError = None
UCXConnectionResetError = None
UCXFirstLinkFailureError = None
UCXLastLinkFailureError = None
UCXFirstEndpointFailureError = None
UCXEndpointTimeoutError = None
UCXLastEndpointFailureError = None

UCXCloseError = None
UCXConfigError = None

# Legacy names
UCXCanceled = None
UCXMsgTruncated = None


def _create_exceptions():
    global UCXError

    global UCXNoMessageError
    global UCXNoResourceError
    global UCXIOError
    global UCXNoMemoryError
    global UCXInvalidParamError
    global UCXUnreachableError
    global UCXInvalidAddrError
    global UCXNotImplementedError
    global UCXMessageTruncatedError
    global UCXNoProgressError
    global UCXBufferTooSmallError
    global UCXNoElemError
    global UCXSomeConnectsFailedError
    global UCXNoDeviceError
    global UCXBusyError
    global UCXCanceledError
    global UCXShmemSegmentError
    global UCXAlreadyExistsError
    global UCXOutOfRangeError
    global UCXTimedOutError
    global UCXExceedsLimitError
    global UCXUnsupportedError
    global UCXRejectedError
    global UCXNotConnectedError
    global UCXConnectionResetError
    global UCXFirstLinkFailureError
    global UCXLastLinkFailureError
    global UCXFirstEndpointFailureError
    global UCXEndpointTimeoutError
    global UCXLastEndpointFailureError

    global UCXCloseError
    global UCXConfigError

    # Legacy names
    global UCXCanceled
    global UCXMsgTruncated

    create_exceptions()

    UCXError = <object>UCXXError

    UCXNoMessageError = <object>UCXXNoMessageError
    UCXNoResourceError = <object>UCXXNoResourceError
    UCXIOError = <object>UCXXIOError
    UCXNoMemoryError = <object>UCXXNoMemoryError
    UCXInvalidParamError = <object>UCXXInvalidParamError
    UCXUnreachableError = <object>UCXXUnreachableError
    UCXInvalidAddrError = <object>UCXXInvalidAddrError
    UCXNotImplementedError = <object>UCXXNotImplementedError
    UCXMessageTruncatedError = <object>UCXXMessageTruncatedError
    UCXNoProgressError = <object>UCXXNoProgressError
    UCXBufferTooSmallError = <object>UCXXBufferTooSmallError
    UCXNoElemError = <object>UCXXNoElemError
    UCXSomeConnectsFailedError = <object>UCXXSomeConnectsFailedError
    UCXNoDeviceError = <object>UCXXNoDeviceError
    UCXBusyError = <object>UCXXBusyError
    UCXCanceledError = <object>UCXXCanceledError
    UCXShmemSegmentError = <object>UCXXShmemSegmentError
    UCXAlreadyExistsError = <object>UCXXAlreadyExistsError
    UCXOutOfRangeError = <object>UCXXOutOfRangeError
    UCXTimedOutError = <object>UCXXTimedOutError
    UCXExceedsLimitError = <object>UCXXExceedsLimitError
    UCXUnsupportedError = <object>UCXXUnsupportedError
    UCXRejectedError = <object>UCXXRejectedError
    UCXNotConnectedError = <object>UCXXNotConnectedError
    UCXConnectionResetError = <object>UCXXConnectionResetError
    UCXFirstLinkFailureError = <object>UCXXFirstLinkFailureError
    UCXLastLinkFailureError = <object>UCXXLastLinkFailureError
    UCXFirstEndpointFailureError = <object>UCXXFirstEndpointFailureError
    UCXEndpointTimeoutError = <object>UCXXEndpointTimeoutError
    UCXLastEndpointFailureError = <object>UCXXLastEndpointFailureError

    UCXCloseError = <object>UCXXCloseError
    UCXConfigError = <object>UCXXConfigError

    # Define legacy names
    # TODO: Deprecate and remove
    UCXCanceled = UCXCanceledError
    UCXMsgTruncated = UCXMessageTruncatedError


###############################################################################
#                                   Types                                     #
###############################################################################

class Feature(enum.Enum):
    TAG = UCP_FEATURE_TAG
    RMA = UCP_FEATURE_RMA
    AMO32 = UCP_FEATURE_AMO32
    AMO64 = UCP_FEATURE_AMO64
    WAKEUP = UCP_FEATURE_WAKEUP
    STREAM = UCP_FEATURE_STREAM
    AM = UCP_FEATURE_AM


class PythonRequestNotifierWaitState(enum.Enum):
    Ready = RequestNotifierWaitState.Ready
    Timeout = RequestNotifierWaitState.Timeout
    Shutdown = RequestNotifierWaitState.Shutdown


###############################################################################
#                                   Classes                                   #
###############################################################################

cdef class UCXConfig():
    cdef:
        unique_ptr[Config] _config
        bint _enable_python_future
        dict _cb_data

    def __init__(self, ConfigMap user_options=ConfigMap()) -> None:
        # TODO: Replace unique_ptr by stack object. Rule-of-five is not allowed
        # by Config, and Cython seems not to handle constructors without moving
        # in `__init__`.
        self._config = move(make_unique[Config](user_options))

    def __dealloc__(self) -> None:
        with nogil:
            self._config.reset()

    @property
    def config(self) -> dict:
        cdef ConfigMap config_map = self._config.get().get()
        return {
            item.first.decode("utf-8"): item.second.decode("utf-8")
            for item in config_map
        }

    def get(self) -> dict:
        warnings.warn(
            "UCXConfig.get() is deprecated and will soon be removed, "
            "use the UCXConfig.config property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.config


cdef class UCXContext():
    """Python representation of `ucp_context_h`

    Parameters
    ----------
    config_dict: Mapping[str, str]
        UCX options such as "MEMTYPE_CACHE=n" and "SEG_SIZE=3M"
    feature_flags: Iterable[Feature]
        Tuple of UCX feature flags
    """
    cdef:
        shared_ptr[Context] _context
        dict _config

    def __init__(
        self,
        dict config_dict=None,
        tuple feature_flags=(
            Feature.TAG,
            Feature.WAKEUP,
            Feature.STREAM,
            Feature.AM,
            Feature.RMA
        )
    ) -> None:
        cdef ConfigMap cpp_config_in, cpp_config_out
        cdef dict context_config

        if config_dict is None:
            config_dict = {}

        for k, v in config_dict.items():
            cpp_config_in[k.encode("utf-8")] = v.encode("utf-8")
        cdef uint64_t feature_flags_uint = functools.reduce(
            lambda x, y: x | y.value, feature_flags, 0
        )

        with nogil:
            self._context = createContext(cpp_config_in, feature_flags_uint)
            cpp_config_out = self._context.get().getConfig()

        context_config = cpp_config_out

        self._config = {
            k.decode("utf-8"): v.decode("utf-8") for k, v in context_config.items()
        }

        logger.info("UCP initiated using config: ")
        for k, v in self._config.items():
            logger.info(f"  {k}, {v}")

    def __dealloc__(self) -> None:
        with nogil:
            self._context.reset()

    @property
    def config(self) -> dict:
        return self._config

    @property
    def feature_flags(self) -> int:
        return int(self._context.get().getFeatureFlags())

    @property
    def cuda_support(self) -> bool:
        return bool(self._context.get().hasCudaSupport())

    @property
    def handle(self) -> int:
        cdef ucp_context_h handle

        with nogil:
            handle = self._context.get().getHandle()

        return int(<uintptr_t>handle)

    @property
    def info(self) -> str:
        cdef Context* ucxx_context
        cdef string info

        with nogil:
            ucxx_context = self._context.get()
            info = ucxx_context.getInfo()

        return info.decode("utf-8")

    cpdef dict get_config(self):
        warnings.warn(
            "UCXContext.get_config() is deprecated and will soon be removed, "
            "use the UCXContext.config property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.config


cdef class UCXAddress():
    cdef:
        shared_ptr[Address] _address
        size_t _length
        ucp_address_t *_handle
        string _string

    def __init__(self) -> None:
        raise TypeError("UCXListener cannot be instantiated directly.")

    def __dealloc__(self) -> None:
        with nogil:
            self._handle = NULL
            self._address.reset()

    @classmethod
    def create_from_worker(cls, UCXWorker worker) -> UCXAddress:
        cdef UCXAddress address = UCXAddress.__new__(UCXAddress)

        with nogil:
            address._address = worker._worker.get().getAddress()
            address._handle = address._address.get().getHandle()
            address._length = address._address.get().getLength()
            address._string = address._address.get().getString()

        return address

    @classmethod
    def create_from_string(cls, string address_str) -> UCXAddress:
        cdef UCXAddress address = UCXAddress.__new__(UCXAddress)
        cdef string cpp_address_str = address_str

        with nogil:
            address._address = createAddressFromString(cpp_address_str)
            address._handle = address._address.get().getHandle()
            address._length = address._address.get().getLength()
            address._string = address._address.get().getString()

        return address

    @classmethod
    def create_from_buffer(cls, bytes buffer) -> UCXAddress:
        cdef string address_str

        buf = Array(buffer)
        assert buf.c_contiguous

        address_str = string(<char*>buf.ptr, <size_t>buf.nbytes)

        return UCXAddress.create_from_string(address_str)

    # For old UCX-Py API compatibility
    @classmethod
    def from_worker(cls, UCXWorker worker) -> UCXAddress:
        warnings.warn(
            "UCXAddress.from_worker() is deprecated and will soon be removed, "
            "use UCXAddress.create_from_worker() instead",
            FutureWarning,
            stacklevel=2,
        )

        return cls.create_from_worker(worker)

    @property
    def address(self) -> int:
        return int(<uintptr_t>self._handle)

    @property
    def length(self) -> int:
        return int(self._length)

    @property
    def string(self) -> bytes:
        return bytes(self._string)

    def __getbuffer__(self, Py_buffer *buffer, int flags) -> None:
        if bool(flags & PyBUF_WRITABLE):
            raise BufferError("Requested writable view on readonly data")
        buffer.buf = self._handle
        buffer.len = self._length
        buffer.obj = self
        buffer.readonly = True
        buffer.itemsize = 1
        if bool(flags & PyBUF_FORMAT):
            buffer.format = b"B"
        else:
            buffer.format = NULL
        buffer.ndim = 1
        if bool(flags & PyBUF_ND):
            buffer.shape = &buffer.len
        else:
            buffer.shape = NULL
        buffer.strides = NULL
        buffer.suboffsets = NULL
        buffer.internal = NULL

    def __releasebuffer__(self, Py_buffer *buffer) -> None:
        pass

    def __reduce__(self) -> tuple:
        return (UCXAddress.create_from_buffer, (self.string,))

    def __hash__(self) -> int:
        return hash(bytes(self.string))


cdef void _generic_callback(void *args) with gil:
    """Callback function called when UCXEndpoint closes or errors"""
    cdef dict cb_data = <dict> args

    try:
        cb_data['cb_func'](
            *cb_data['cb_args'],
            **cb_data['cb_kwargs']
        )
    except Exception as e:
        pass


cdef class UCXWorker():
    """Python representation of `ucp_worker_h`"""
    cdef:
        shared_ptr[Worker] _worker
        dict _progress_thread_start_cb_data
        bint _enable_delayed_submission
        bint _enable_python_future
        uint64_t _context_feature_flags

    def __init__(
            self,
            UCXContext context,
            bint enable_delayed_submission=False,
            bint enable_python_future=False,
    ) -> None:
        cdef bint ucxx_enable_delayed_submission = enable_delayed_submission
        cdef bint ucxx_enable_python_future = enable_python_future
        cdef AmAllocatorType rmm_am_allocator

        self._context_feature_flags = <uint64_t>(context.feature_flags)

        with nogil:
            self._worker = createPythonWorker(
                context._context,
                ucxx_enable_delayed_submission,
                ucxx_enable_python_future,
            )
            self._enable_delayed_submission = (
                self._worker.get().isDelayedRequestSubmissionEnabled()
            )
            self._enable_python_future = self._worker.get().isFutureEnabled()

            if self._context_feature_flags & UCP_FEATURE_AM:
                rmm_am_allocator = <AmAllocatorType>(&_rmm_am_allocator)
                self._worker.get().registerAmAllocator(
                    UCS_MEMORY_TYPE_CUDA, rmm_am_allocator
                )

    def __dealloc__(self) -> None:
        with nogil:
            self._worker.reset()

    @property
    def handle(self) -> int:
        cdef ucp_worker_h handle

        with nogil:
            handle = self._worker.get().getHandle()

        return int(<uintptr_t>handle)

    @property
    def ucxx_ptr(self) -> int:
        cdef Worker* worker

        with nogil:
            worker = self._worker.get()

        return int(<uintptr_t>worker)

    @property
    def info(self) -> str:
        cdef Worker* ucxx_worker
        cdef string info

        with nogil:
            ucxx_worker = self._worker.get()
            info = ucxx_worker.getInfo()

        return info.decode("utf-8")

    @property
    def address(self) -> UCXAddress:
        return UCXAddress.create_from_worker(self)

    @property
    def enable_delayed_submission(self) -> bool:
        return self._enable_delayed_submission

    @property
    def enable_python_future(self) -> bool:
        return self._enable_python_future

    def get_address(self) -> UCXAddress:
        warnings.warn(
            "UCXWorker.get_address() is deprecated and will soon be removed, "
            "use the UCXWorker.address property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.address

    def create_endpoint_from_hostname(
            self,
            str ip_address,
            uint16_t port,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        return UCXEndpoint.create(self, ip_address, port, endpoint_error_handling)

    def create_endpoint_from_worker_address(
            self,
            UCXAddress address,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        return UCXEndpoint.create_from_worker_address(
            self, address, endpoint_error_handling
        )

    def init_blocking_progress_mode(self) -> None:
        with nogil:
            self._worker.get().initBlockingProgressMode()

    def arm(self) -> bool:
        cdef bint armed

        with nogil:
            armed = self._worker.get().arm()

        return armed

    @property
    def epoll_file_descriptor(self) -> int:
        cdef int epoll_file_descriptor = 0

        with nogil:
            epoll_file_descriptor = self._worker.get().getEpollFileDescriptor()

        return epoll_file_descriptor

    def progress(self) -> None:
        with nogil:
            self._worker.get().progress()

    def progress_once(self) -> bool:
        cdef bint progress_made

        with nogil:
            progress_made = self._worker.get().progressOnce()

        return progress_made

    def progress_worker_event(self, int epoll_timeout=-1) -> None:
        cdef int ucxx_epoll_timeout = epoll_timeout

        with nogil:
            self._worker.get().progressWorkerEvent(ucxx_epoll_timeout)

    def start_progress_thread(
        self,
        bint polling_mode=False,
        int epoll_timeout=-1
    ) -> None:
        cdef int ucxx_epoll_timeout = epoll_timeout

        with nogil:
            self._worker.get().startProgressThread(
                polling_mode, epoll_timeout=ucxx_epoll_timeout
            )

    def stop_progress_thread(self) -> None:
        with nogil:
            self._worker.get().stopProgressThread()

    def cancel_inflight_requests(
        self,
        uint64_t period=0,
        uint64_t max_attempts=1
    ) -> int:
        cdef uint64_t c_period = period
        cdef uint64_t c_max_attempts = max_attempts
        cdef size_t num_canceled

        with nogil:
            num_canceled = self._worker.get().cancelInflightRequests(
                c_period, c_max_attempts
            )

        return num_canceled

    def tag_probe(self, UCXXTag tag) -> bool:
        cdef bint tag_matched
        cdef Tag cpp_tag = <Tag><size_t>tag.value

        with nogil:
            tag_matched = self._worker.get().tagProbe(cpp_tag)

        return tag_matched

    def set_progress_thread_start_callback(
            self, cb_func, tuple cb_args=None, dict cb_kwargs=None
    ) -> None:
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}

        self._progress_thread_start_cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }

        cdef function[void(void*)]* func_generic_callback = (
            new function[void(void*)](_generic_callback)
        )
        with nogil:
            self._worker.get().setProgressThreadStartCallback(
                deref(func_generic_callback), <void*>self._progress_thread_start_cb_data
            )
        del func_generic_callback

    def stop_request_notifier_thread(self) -> None:
        with nogil:
            self._worker.get().stopRequestNotifierThread()

    def wait_request_notifier(
            self,
            uint64_t
            period_ns=0
    ) -> PythonRequestNotifierWaitState:
        cdef RequestNotifierWaitState state
        cdef uint64_t p = period_ns

        with nogil:
            state = self._worker.get().waitRequestNotifier(p)

        return PythonRequestNotifierWaitState(state)

    def run_request_notifier(self) -> None:
        with nogil:
            self._worker.get().runRequestNotifier()

    def populate_python_futures_pool(self) -> None:
        with nogil:
            self._worker.get().populateFuturesPool()

    def is_delayed_submission_enabled(self) -> bool:
        warnings.warn(
            "UCXWorker.is_delayed_submission_enabled() is deprecated and will soon "
            "be removed, use the UCXWorker.enable_delayed_submission property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.enable_delayed_submission

    def is_python_future_enabled(self) -> bool:
        warnings.warn(
            "UCXWorker.is_python_future_enabled() is deprecated and will soon be "
            "removed, use the UCXWorker.enable_python_future property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.enable_python_future

    def tag_recv(
        self,
        Array arr,
        UCXXTag tag,
        UCXXTagMask tag_mask = UCXXTagMaskFull
    ) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req
        cdef Tag cpp_tag = <Tag><size_t>tag.value
        cdef TagMask cpp_tag_mask = <TagMask><size_t>tag_mask.value

        if not self._context_feature_flags & Feature.TAG.value:
            raise ValueError("UCXContext must be created with `Feature.TAG`")

        with nogil:
            req = self._worker.get().tagRecv(
                buf,
                nbytes,
                cpp_tag,
                cpp_tag_mask,
                self._enable_python_future
            )

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)


cdef class UCXRequest():
    cdef:
        shared_ptr[Request] _request
        bint _enable_python_future
        bint _completed

    def __init__(self, uintptr_t shared_ptr_request, bint enable_python_future) -> None:
        self._request = deref(<shared_ptr[Request] *> shared_ptr_request)
        self._enable_python_future = enable_python_future
        self._completed = False

    def __dealloc__(self) -> None:
        with nogil:
            self._request.get().cancel()
            self._request.reset()

    @property
    def completed(self) -> bool:
        cdef bint completed

        if self._completed is True:
            return True

        with nogil:
            completed = self._request.get().isCompleted()

        return completed

    @property
    def status(self) -> ucs_status_t:
        cdef ucs_status_t status

        with nogil:
            status = self._request.get().getStatus()

        return status

    @property
    def future(self) -> object:
        cdef PyObject* future_ptr

        with nogil:
            future_ptr = <PyObject*>self._request.get().getFuture()

        return <object>future_ptr

    @property
    def recv_buffer(self) -> None|np.ndarray|DeviceBuffer:
        cdef shared_ptr[Buffer] buf
        cdef BufferType bufType

        with nogil:
            buf = self._request.get().getRecvBuffer()
            bufType = buf.get().getType() if buf != nullptr else BufferType.Invalid

        # If buf == NULL, it's not allocated by the request but rather the user
        if buf == NULL:
            return None
        elif bufType == BufferType.RMM:
            return _get_rmm_buffer(<uintptr_t><void*>buf.get())
        elif bufType == BufferType.Host:
            return _get_host_buffer(<uintptr_t><void*>buf.get())

    def is_completed(self) -> bool:
        warnings.warn(
            "UCXRequest.is_completed() is deprecated and will soon be removed, "
            "use the UCXRequest.completed property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.completed

    def get_status(self) -> ucs_status_t:
        warnings.warn(
            "UCXRequest.get_status() is deprecated and will soon be removed, "
            "use the UCXRequest.status property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.status

    def check_error(self) -> None:
        with nogil:
            self._request.get().checkError()

    async def wait_yield(self) -> None:
        while True:
            if self.completed:
                return self.check_error()
            await asyncio.sleep(0)

    def get_future(self) -> object:
        warnings.warn(
            "UCXRequest.get_future() is deprecated and will soon be removed, "
            "use the UCXRequest.future property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.future

    async def wait(self) -> None:
        if self._enable_python_future:
            await self.future
        else:
            await self.wait_yield()

    def get_recv_buffer(self) -> None|np.ndarray|DeviceBuffer:
        warnings.warn(
            "UCXRequest.get_recv_buffer() is deprecated and will soon be removed, "
            "use the UCXRequest.recv_buffer property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.recv_buffer


cdef class UCXBufferRequest:
    cdef:
        BufferRequestPtr _buffer_request
        bint _enable_python_future

    def __init__(
            self,
            uintptr_t shared_ptr_buffer_request,
            bint enable_python_future
    ) -> None:
        self._buffer_request = deref(<BufferRequestPtr *> shared_ptr_buffer_request)
        self._enable_python_future = enable_python_future

    def __dealloc__(self) -> None:
        with nogil:
            self._buffer_request.reset()

    @property
    def request(self) -> UCXRequest:
        return UCXRequest(
            <uintptr_t><void*>&self._buffer_request.get().request,
            self._enable_python_future,
        )

    @property
    def py_buffer(self) -> None|np.ndarray|DeviceBuffer:
        cdef shared_ptr[Buffer] buf
        cdef BufferType bufType

        with nogil:
            buf = self._buffer_request.get().buffer
            bufType = buf.get().getType() if buf != nullptr else BufferType.Invalid

        # If buf == NULL, it holds a header
        if buf == NULL:
            return None
        elif bufType == BufferType.RMM:
            return _get_rmm_buffer(<uintptr_t><void*>buf.get())
        elif bufType == BufferType.Host:
            return _get_host_buffer(<uintptr_t><void*>buf.get())

    def get_request(self) -> UCXRequest:
        warnings.warn(
            "UCXBufferRequest.get_request() is deprecated and will soon be removed, "
            "use the UCXBufferRequest.request property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.request

    def get_py_buffer(self) -> None|np.ndarray|DeviceBuffer:
        warnings.warn(
            "UCXBufferRequest.get_py_buffer() is deprecated and will soon be removed, "
            "use the UCXBufferRequest.py_buffer property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.py_buffer


cdef class UCXBufferRequests:
    cdef:
        RequestTagMultiPtr _ucxx_request_tag_multi
        bint _enable_python_future
        bint _completed
        tuple _buffer_requests
        tuple _requests

    def __init__(
        self,
        uintptr_t unique_ptr_buffer_requests,
        bint enable_python_future
    ) -> None:
        self._enable_python_future = enable_python_future
        self._completed = False
        self._requests = tuple()

        self._ucxx_request_tag_multi = (
            deref(<RequestTagMultiPtr *> unique_ptr_buffer_requests)
        )

    def __dealloc__(self) -> None:
        with nogil:
            self._ucxx_request_tag_multi.reset()

    def _populate_requests(self) -> None:
        cdef vector[BufferRequestPtr] requests
        if len(self._requests) == 0:
            requests = deref(self._ucxx_request_tag_multi)._bufferRequests
            total_requests = requests.size()
            self._buffer_requests = tuple([
                UCXBufferRequest(
                    <uintptr_t><void*>&(requests[i]),
                    self._enable_python_future
                )
                for i in range(total_requests)
            ])

            self._requests = tuple([br.request for br in self._buffer_requests])

    @property
    def completed(self) -> bool:
        cdef bint completed

        if self._completed is False:
            with nogil:
                completed = self._ucxx_request_tag_multi.get().isCompleted()
            self._completed = completed

        return self._completed

    @property
    def all_completed(self) -> bool:
        if self._completed is False:
            if self._ucxx_request_tag_multi.get()._isFilled is False:
                return False

            self._populate_requests()

            self._completed = all(
                [r.completed for r in self._requests]
            )

        return self._completed

    @property
    def status(self) -> ucx_status_t:
        cdef ucs_status_t status

        with nogil:
            status = self._ucxx_request_tag_multi.get().getStatus()

        return status

    @property
    def future(self) -> object:
        cdef PyObject* future_ptr

        with nogil:
            future_ptr = <PyObject*>self._ucxx_request_tag_multi.get().getFuture()

        return <object>future_ptr

    @property
    def py_buffers(self) -> tuple[None|np.ndarray|DeviceBuffer, ...]:
        if not self.completed:
            raise RuntimeError("Some requests are not completed yet")

        self._populate_requests()

        py_buffers = [br.py_buffer for br in self._buffer_requests]
        # PyBuffers that are None are headers
        return [b for b in py_buffers if b is not None]

    @property
    def requests(self) -> tuple[UCXRequest, ...]:
        self._populate_requests()
        return self._requests

    def is_completed(self) -> bool:
        warnings.warn(
            "UCXBufferRequests.is_completed() is deprecated and will soon be removed, "
            "use the UCXBufferRequests.completed property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.completed

    def is_completed_all(self) -> bool:
        warnings.warn(
            "UCXBufferRequests.is_completed_all() is deprecated and will soon be "
            "removed, use the UCXBufferRequests.all_completed property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.all_completed

    def check_error(self) -> None:
        with nogil:
            self._ucxx_request_tag_multi.get().checkError()

    def get_status(self) -> ucs_status_t:
        warnings.warn(
            "UCXBufferRequests.get_status() is deprecated and will soon be removed, "
            "use the UCXBufferRequests.status property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.status

    async def wait_yield(self) -> None:
        while True:
            if self.completed:
                for r in self._requests:
                    r.check_error()
                return
            await asyncio.sleep(0)

    def get_future(self) -> object:
        warnings.warn(
            "UCXBufferRequests.get_future() is deprecated and will soon be removed, "
            "use the UCXBufferRequests.future property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.future

    async def wait(self) -> None:
        if self._enable_python_future:
            await self.future
        else:
            await self.wait_yield()

    def get_requests(self) -> tuple[UCXRequest, ...]:
        warnings.warn(
            "UCXBufferRequests.get_requests() is deprecated and will soon be removed, "
            "use the UCXBufferRequests.requests property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.requests

    def get_py_buffers(self) -> tuple[None|np.ndarray|DeviceBuffer, ...]:
        warnings.warn(
            "UCXBufferRequests.get_py_buffers() is deprecated and will soon be "
            "removed, use the UCXBufferRequests.py_buffers property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.py_buffers


cdef void _endpoint_close_callback(ucs_status_t status, shared_ptr[void] args) with gil:
    """Callback function called when UCXEndpoint closes or errors"""
    cdef shared_ptr[uintptr_t] cb_data_ptr = static_pointer_cast[uintptr_t, void](args)
    cdef dict cb_data = <dict><void*>cb_data_ptr.get()[0]

    try:
        cb_data['cb_func'](
            *cb_data['cb_args'],
            **cb_data['cb_kwargs']
        )
    except Exception as e:
        logger.error(f"{type(e)} when calling endpoint close callback: {e}")


cdef class UCXEndpoint():
    cdef:
        shared_ptr[Endpoint] _endpoint
        uint64_t _context_feature_flags
        bint _cuda_support
        bint _enable_python_future
        dict _close_cb_data
        shared_ptr[uintptr_t] _close_cb_data_ptr

    def __init__(self) -> None:
        raise TypeError("UCXListener cannot be instantiated directly.")

    def __dealloc__(self) -> None:
        self.remove_close_callback()

        with nogil:
            self._endpoint.reset()

    @classmethod
    def create(
            cls,
            UCXWorker worker,
            str ip_address,
            uint16_t port,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        cdef UCXEndpoint endpoint = UCXEndpoint.__new__(UCXEndpoint)
        cdef shared_ptr[Context] ucxx_context
        cdef string addr = ip_address.encode("utf-8")

        endpoint._enable_python_future = worker.enable_python_future

        with nogil:
            ucxx_context = dynamic_pointer_cast[Context, Component](
                worker._worker.get().getParent()
            )

            endpoint._context_feature_flags = ucxx_context.get().getFeatureFlags()
            endpoint._cuda_support = ucxx_context.get().hasCudaSupport()
            endpoint._endpoint = worker._worker.get().createEndpointFromHostname(
                addr, port, endpoint_error_handling
            )

        return endpoint

    @classmethod
    def create_from_conn_request(
            cls,
            UCXListener listener,
            uintptr_t conn_request,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        cdef UCXEndpoint endpoint = UCXEndpoint.__new__(UCXEndpoint)
        cdef shared_ptr[Context] ucxx_context
        cdef shared_ptr[Worker] ucxx_worker

        endpoint._enable_python_future = listener.enable_python_future

        with nogil:
            ucxx_worker = dynamic_pointer_cast[Worker, Component](
                listener._listener.get().getParent()
            )
            ucxx_context = dynamic_pointer_cast[Context, Component](
                ucxx_worker.get().getParent()
            )

            endpoint._context_feature_flags = ucxx_context.get().getFeatureFlags()
            endpoint._cuda_support = ucxx_context.get().hasCudaSupport()
            endpoint._endpoint = listener._listener.get().createEndpointFromConnRequest(
                <ucp_conn_request_h>conn_request, endpoint_error_handling
            )

        return endpoint

    @classmethod
    def create_from_worker_address(
            cls,
            UCXWorker worker,
            UCXAddress address,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        cdef UCXEndpoint endpoint = UCXEndpoint.__new__(UCXEndpoint)
        cdef shared_ptr[Context] ucxx_context
        cdef shared_ptr[Address] ucxx_address = address._address

        endpoint._enable_python_future = worker.enable_python_future

        with nogil:
            ucxx_context = dynamic_pointer_cast[Context, Component](
                worker._worker.get().getParent()
            )

            endpoint._context_feature_flags = ucxx_context.get().getFeatureFlags()
            endpoint._cuda_support = ucxx_context.get().hasCudaSupport()
            endpoint._endpoint = worker._worker.get().createEndpointFromWorkerAddress(
                ucxx_address, endpoint_error_handling
            )

        return endpoint

    @property
    def handle(self) -> int:
        cdef ucp_ep_h handle

        with nogil:
            handle = self._endpoint.get().getHandle()

        return int(<uintptr_t>handle)

    @property
    def ucxx_ptr(self) -> int:
        cdef Endpoint* endpoint

        with nogil:
            endpoint = self._endpoint.get()

        return int(<uintptr_t>endpoint)

    @property
    def worker_handle(self) -> int:
        cdef ucp_worker_h handle

        with nogil:
            handle = self._endpoint.get().getWorker().get().getHandle()

        return int(<uintptr_t>handle)

    @property
    def ucxx_worker_ptr(self) -> int:
        cdef Worker* worker

        with nogil:
            worker = self._endpoint.get().getWorker().get()

        return int(<uintptr_t>worker)

    @property
    def alive(self) -> bool:
        cdef bint alive

        with nogil:
            alive = self._endpoint.get().isAlive()

        return alive

    def close(self) -> None:
        cdef shared_ptr[Request] req

        with nogil:
            req = self._endpoint.get().close(
                self._enable_python_future
            )

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def close_blocking(self, uint64_t period=0, uint64_t max_attempts=1) -> None:
        cdef uint64_t c_period = period
        cdef uint64_t c_max_attempts = max_attempts

        with nogil:
            self._endpoint.get().closeBlocking(c_period, c_max_attempts)

    def am_probe(self) -> bool:
        cdef ucp_ep_h handle
        cdef shared_ptr[Worker] worker
        cdef bint ep_matched

        with nogil:
            handle = self._endpoint.get().getHandle()
            worker = self._endpoint.get().getWorker()
            ep_matched = worker.get().amProbe(handle)

        return ep_matched

    def am_send(self, Array arr) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef bint cuda_array = arr.cuda
        cdef shared_ptr[Request] req

        if not self._context_feature_flags & Feature.AM.value:
            raise ValueError("UCXContext must be created with `Feature.AM`")

        with nogil:
            req = self._endpoint.get().amSend(
                buf,
                nbytes,
                UCS_MEMORY_TYPE_CUDA if cuda_array else UCS_MEMORY_TYPE_HOST,
                nullopt,
                self._enable_python_future
            )

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def am_recv(self) -> UCXRequest:
        cdef shared_ptr[Request] req

        if not self._context_feature_flags & Feature.AM.value:
            raise ValueError("UCXContext must be created with `Feature.AM`")

        with nogil:
            req = self._endpoint.get().amRecv(self._enable_python_future)

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def stream_send(self, Array arr) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req

        if not self._context_feature_flags & Feature.STREAM.value:
            raise ValueError("UCXContext must be created with `Feature.STREAM`")
        if arr.cuda and not self._cuda_support:
            raise ValueError(
                "UCX is not configured with CUDA support, please ensure that the "
                "available UCX on your environment is built against CUDA and that "
                "`cuda` or `cuda_copy` are present in `UCX_TLS` or that it is using "
                "the default `UCX_TLS=all`."
            )

        with nogil:
            req = self._endpoint.get().streamSend(
                buf,
                nbytes,
                self._enable_python_future
            )

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def stream_recv(self, Array arr) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req

        if not self._context_feature_flags & Feature.STREAM.value:
            raise ValueError("UCXContext must be created with `Feature.STREAM`")
        if arr.cuda and not self._cuda_support:
            raise ValueError(
                "UCX is not configured with CUDA support, please ensure that the "
                "available UCX on your environment is built against CUDA and that "
                "`cuda` or `cuda_copy` are present in `UCX_TLS` or that it is using "
                "the default `UCX_TLS=all`."
            )

        with nogil:
            req = self._endpoint.get().streamRecv(
                buf,
                nbytes,
                self._enable_python_future
            )

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def tag_send(self, Array arr, UCXXTag tag) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req
        cdef Tag cpp_tag = <Tag><size_t>tag.value

        if not self._context_feature_flags & Feature.TAG.value:
            raise ValueError("UCXContext must be created with `Feature.TAG`")
        if arr.cuda and not self._cuda_support:
            raise ValueError(
                "UCX is not configured with CUDA support, please ensure that the "
                "available UCX on your environment is built against CUDA and that "
                "`cuda` or `cuda_copy` are present in `UCX_TLS` or that it is using "
                "the default `UCX_TLS=all`."
            )

        with nogil:
            req = self._endpoint.get().tagSend(
                buf,
                nbytes,
                cpp_tag,
                self._enable_python_future
            )

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def tag_recv(
        self,
        Array arr,
        UCXXTag tag,
        UCXXTagMask tag_mask=UCXXTagMaskFull
    ) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req
        cdef Tag cpp_tag = <Tag><size_t>tag.value
        cdef TagMask cpp_tag_mask = <TagMask><size_t>tag_mask.value

        if not self._context_feature_flags & Feature.TAG.value:
            raise ValueError("UCXContext must be created with `Feature.TAG`")
        if arr.cuda and not self._cuda_support:
            raise ValueError(
                "UCX is not configured with CUDA support, please ensure that the "
                "available UCX on your environment is built against CUDA and that "
                "`cuda` or `cuda_copy` are present in `UCX_TLS` or that it is using "
                "the default `UCX_TLS=all`."
            )

        with nogil:
            req = self._endpoint.get().tagRecv(
                buf,
                nbytes,
                cpp_tag,
                cpp_tag_mask,
                self._enable_python_future
            )

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def tag_send_multi(self, tuple arrays, UCXXTag tag) -> UCXBufferRequests:
        cdef vector[void*] v_buffer
        cdef vector[size_t] v_size
        cdef vector[int] v_is_cuda
        cdef shared_ptr[Request] ucxx_buffer_requests
        cdef Tag cpp_tag = <Tag><size_t>tag.value

        for arr in arrays:
            if not isinstance(arr, Array):
                raise ValueError(
                    "All elements of the `arrays` should be of `Array` type"
                )
            if arr.cuda and not self._cuda_support:
                raise ValueError(
                    "UCX is not configured with CUDA support, please ensure that the "
                    "available UCX on your environment is built against CUDA and that "
                    "`cuda` or `cuda_copy` are present in `UCX_TLS` or that it is "
                    "using the default `UCX_TLS=all`."
                )

        for arr in arrays:
            v_buffer.push_back(<void*><uintptr_t>arr.ptr)
            v_size.push_back(arr.nbytes)
            v_is_cuda.push_back(arr.cuda)

        with nogil:
            ucxx_buffer_requests = self._endpoint.get().tagMultiSend(
                v_buffer,
                v_size,
                v_is_cuda,
                cpp_tag,
                self._enable_python_future,
            )

        return UCXBufferRequests(
            <uintptr_t><void*>&ucxx_buffer_requests, self._enable_python_future,
        )

    def tag_recv_multi(
        self,
        UCXXTag tag,
        UCXXTagMask tag_mask = UCXXTagMaskFull,
    ) -> UCXBufferRequests:
        cdef shared_ptr[Request] ucxx_buffer_requests
        cdef Tag cpp_tag = <Tag><size_t>tag.value
        cdef TagMask cpp_tag_mask = <TagMask><size_t>tag_mask.value

        with nogil:
            ucxx_buffer_requests = self._endpoint.get().tagMultiRecv(
                cpp_tag, cpp_tag_mask, self._enable_python_future
            )

        return UCXBufferRequests(
            <uintptr_t><void*>&ucxx_buffer_requests, self._enable_python_future,
        )

    def is_alive(self) -> bool:
        warnings.warn(
            "UCXEndpoint.is_alive() is deprecated and will soon be removed, "
            "use the UCXEndpoint.alive property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.alive

    def raise_on_error(self) -> None:
        with nogil:
            self._endpoint.get().raiseOnError()

    def set_close_callback(
            self,
            cb_func,
            tuple cb_args=None,
            dict cb_kwargs=None
    ) -> None:
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}

        self._close_cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }
        self._close_cb_data_ptr = make_shared[uintptr_t](
            <uintptr_t><void*>self._close_cb_data
        )

        cdef function[void(ucs_status_t, shared_ptr[void])]* func_close_callback = (
            new function[void(ucs_status_t, shared_ptr[void])](_endpoint_close_callback)
        )
        with nogil:
            self._endpoint.get().setCloseCallback(
                deref(func_close_callback),
                static_pointer_cast[void, uintptr_t](self._close_cb_data_ptr)
            )
        del func_close_callback

    def remove_close_callback(self) -> None:
        cdef Endpoint* endpoint

        with nogil:
            # Unset close callback, in case the Endpoint error callback runs
            # after the Python object has been destroyed.
            # Cast explicitly to prevent Cython `Cannot assign type ...` errors.
            endpoint = self._endpoint.get()
            if endpoint != nullptr:
                endpoint.setCloseCallback(
                    <function[void (ucs_status_t, shared_ptr[void]) except *]>nullptr,
                    <shared_ptr[void]>nullptr,
                )


cdef void _listener_callback(ucp_conn_request_h conn_request, void *args) with gil:
    """Callback function used by UCXListener"""
    cdef dict cb_data = <dict> args

    try:
        cb_data['cb_func'](
            (
                cb_data['listener']().create_endpoint_from_conn_request(
                    int(<uintptr_t>conn_request), True
                ) if 'listener' in cb_data else
                int(<uintptr_t>conn_request)
            ),
            *cb_data['cb_args'],
            **cb_data['cb_kwargs']
        )
    except Exception as e:
        logger.error(f"{type(e)} when calling listener callback: {e}")


cdef class UCXListener():
    cdef:
        shared_ptr[Listener] _listener
        bint _enable_python_future
        dict _cb_data
        object __weakref__

    def __init__(self) -> None:
        raise TypeError("UCXListener cannot be instantiated directly.")

    def __dealloc__(self) -> None:
        with nogil:
            self._listener.reset()

    @classmethod
    def create(
            cls,
            UCXWorker worker,
            uint16_t port,
            cb_func,
            tuple cb_args=None,
            dict cb_kwargs=None,
            bint deliver_endpoint=False,
    ) -> UCXListener:
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}

        cdef UCXListener listener = UCXListener.__new__(UCXListener)
        cdef ucp_listener_conn_callback_t listener_cb = (
            <ucp_listener_conn_callback_t>_listener_callback
        )
        cdef dict cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }
        if deliver_endpoint is True:
            cb_data["listener"] = weakref.ref(listener)

        listener._cb_data = cb_data
        listener._enable_python_future = worker.enable_python_future

        with nogil:
            listener._listener = worker._worker.get().createListener(
                port, listener_cb, <void*>listener._cb_data
            )

        return listener

    @property
    def port(self) -> int:
        cdef uint16_t port

        with nogil:
            port = self._listener.get().getPort()

        return port

    @property
    def ip(self) -> str:
        cdef string ip

        with nogil:
            ip = self._listener.get().getIp()

        return ip.decode("utf-8")

    @property
    def enable_python_future(self) -> bool:
        return self._enable_python_future

    def create_endpoint_from_conn_request(
            self,
            uintptr_t conn_request,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        return UCXEndpoint.create_from_conn_request(
            self, conn_request, endpoint_error_handling,
        )

    def is_python_future_enabled(self) -> bool:
        warnings.warn(
            "UCXListener.is_python_future_enabled() is deprecated and will soon be "
            "removed, use the UCXListener.enable_python_future property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.enable_python_future


def get_current_options() -> dict:
    """
    Returns the current UCX options
    if UCX were to be initialized now.
    """
    return UCXConfig().config


def get_ucx_version() -> tuple[int, int, int]:
    cdef unsigned int a, b, c
    ucp_get_version(&a, &b, &c)
    return (a, b, c)
