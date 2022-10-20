# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3

import asyncio
import enum
import functools
import logging

from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp cimport nullptr
from libcpp.functional cimport function
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport dynamic_pointer_cast, shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

import numpy as np

from rmm._lib.device_buffer cimport DeviceBuffer

from . cimport ucxx_api
from .arr cimport Array
from .ucxx_api cimport *

logger = logging.getLogger("ucx")


np.import_array()


cdef ptr_to_ndarray(void* ptr, np.npy_intp N):
    cdef np.ndarray[np.uint8_t, ndim=1, mode="c"] arr = (
        np.PyArray_SimpleNewFromData(1, &N, np.NPY_UINT8, <np.uint8_t*>ptr)
    )
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA)
    return arr


def _get_rmm_buffer(uintptr_t recv_buffer_ptr):
    cdef RMMBuffer* rmm_buffer = <RMMBuffer*>recv_buffer_ptr
    return DeviceBuffer.c_from_unique_ptr(move(rmm_buffer.release()))


def _get_host_buffer(uintptr_t recv_buffer_ptr):
    cdef HostBuffer* host_buffer = <HostBuffer*>recv_buffer_ptr
    cdef size_t size = host_buffer.getSize()
    return ptr_to_ndarray(host_buffer.release(), size)


###############################################################################
#                               Exceptions                                    #
###############################################################################


class UCXBaseException(Exception):
    pass


class UCXError(UCXBaseException):
    pass


class UCXCanceled(UCXBaseException):
    pass


class UCXCloseError(UCXBaseException):
    pass


class UCXConfigError(UCXBaseException):
    pass


class UCXConnectionResetError(UCXBaseException):
    pass


cdef public PyObject* ucxx_error = <PyObject*>UCXError
cdef public PyObject* ucxx_canceled_error = <PyObject*>UCXCanceled
cdef public PyObject* ucxx_config_error = <PyObject*>UCXConfigError
cdef public PyObject* ucxx_connection_reset_error = <PyObject*>UCXConnectionResetError


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
    Ready = UcxxPythonRequestNotifierWaitStateReady
    Timeout = UcxxPythonRequestNotifierWaitStateTimeout
    Shutdown = UcxxPythonRequestNotifierWaitStateShutdown


###############################################################################
#                                   Classes                                   #
###############################################################################

def PythonEnabled():
    cdef int python_enabled
    with nogil:
        python_enabled = UCXX_ENABLE_PYTHON
    return bool(python_enabled)


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
        config_dict={},
        feature_flags=(
            Feature.TAG,
            Feature.WAKEUP,
            Feature.STREAM,
            Feature.AM,
            Feature.RMA
        )
    ):
        cdef ConfigMap cpp_config_in, cpp_config_out
        cdef dict context_config

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

    cpdef dict get_config(self):
        return self._config

    @property
    def handle(self):
        cdef ucp_context_h handle

        with nogil:
            handle = self._context.get().getHandle()

        return int(<uintptr_t>handle)

    cpdef str get_info(self):
        cdef Context* ucxx_context
        cdef string info

        with nogil:
            ucxx_context = self._context.get()
            info = ucxx_context.getInfo()

        return info.decode("utf-8")


cdef class UCXAddress():
    cdef:
        shared_ptr[Address] _address

    def __init__(self, uintptr_t shared_ptr_address):
        self._address = deref(<shared_ptr[Address] *> shared_ptr_address)

    @classmethod
    def create_from_worker(cls, UCXWorker worker):
        cdef shared_ptr[Address] address

        with nogil:
            address = worker._worker.get().getAddress()

        return cls(<uintptr_t><void*>&address)

    # For old UCX-Py API compatibility
    @classmethod
    def from_worker(cls, UCXWorker worker):
        return cls.create_from_worker(worker)

    @property
    def address(self):
        cdef ucp_address_t* address

        with nogil:
            address = self._address.get().getHandle()

        return int(<uintptr_t>address)

    @property
    def length(self):
        cdef size_t getLength

        with nogil:
            length = self._address.get().getLength()

        return int(length)


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
        bint _enable_python_future

    def __init__(
            self,
            UCXContext context,
            enable_delayed_submission=False,
            enable_python_future=False,
    ):
        cdef bint ucxx_enable_delayed_submission = enable_delayed_submission
        cdef bint ucxx_enable_python_future = enable_python_future
        with nogil:
            self._worker = context._context.get().createWorker(
                ucxx_enable_delayed_submission,
                ucxx_enable_python_future,
            )
        self._enable_python_future = PythonEnabled() and enable_python_future

    @property
    def handle(self):
        cdef ucp_worker_h handle

        with nogil:
            handle = self._worker.get().getHandle()

        return int(<uintptr_t>handle)

    def get_address(self):
        return UCXAddress.create_from_worker(self)

    def create_endpoint_from_hostname(
            self,
            str ip_address,
            uint16_t port,
            bint endpoint_error_handling
    ):
        return UCXEndpoint.create(self, ip_address, port, endpoint_error_handling)

    def create_endpoint_from_worker_address(
            self,
            UCXAddress address,
            bint endpoint_error_handling
    ):
        return UCXEndpoint.create_from_worker_address(
            self, address, endpoint_error_handling
        )

    def init_blocking_progress_mode(self):
        with nogil:
            self._worker.get().initBlockingProgressMode()

    def progress(self):
        with nogil:
            self._worker.get().progress()

    def progress_once(self):
        cdef bint progress_made

        with nogil:
            progress_made = self._worker.get().progressOnce()

        return progress_made

    def progress_worker_event(self):
        with nogil:
            self._worker.get().progressWorkerEvent()

    def start_progress_thread(self, bint polling_mode=False):
        with nogil:
            self._worker.get().startProgressThread(polling_mode)

    def stop_progress_thread(self):
        with nogil:
            self._worker.get().stopProgressThread()

    def cancel_inflight_requests(self):
        cdef size_t num_canceled

        with nogil:
            num_canceled = self._worker.get().cancelInflightRequests()

        return num_canceled

    def tag_probe(self, size_t tag):
        cdef bint tag_matched

        with nogil:
            tag_matched = self._worker.get().tagProbe(tag)

        return tag_matched

    def set_progress_thread_start_callback(
            self, cb_func, tuple cb_args=None, dict cb_kwargs=None
    ):
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

    def stop_request_notifier_thread(self):
        with nogil:
            self._worker.get().stopRequestNotifierThread()

    def wait_request_notifier(self, period_ns=0):
        cdef RequestNotifierWaitState state
        cdef uint64_t p = period_ns

        with nogil:
            state = self._worker.get().waitRequestNotifier(p)

        return PythonRequestNotifierWaitState(state)

    def run_request_notifier(self):
        with nogil:
            self._worker.get().runRequestNotifier()

    def populate_python_futures_pool(self):
        with nogil:
            self._worker.get().populatePythonFuturesPool()

    def is_python_future_enabled(self):
        return self._enable_python_future


cdef class UCXRequest():
    cdef:
        shared_ptr[Request] _request
        bint _enable_python_future
        bint _is_completed

    def __init__(self, uintptr_t shared_ptr_request, bint enable_python_future):
        self._request = deref(<shared_ptr[Request] *> shared_ptr_request)
        self._enable_python_future = enable_python_future
        self._is_completed = False

    def is_completed(self, int64_t period_ns=0):
        cdef bint is_completed

        if self._is_completed is True:
            return True

        with nogil:
            is_completed = self._request.get().isCompleted(period_ns)

        return is_completed

    def get_status(self):
        cdef ucs_status_t status

        with nogil:
            status = self._request.get().getStatus()

        return status

    def check_error(self):
        with nogil:
            self._request.get().checkError()

    async def wait_yield(self, period_ns=0):
        while True:
            if self.is_completed(period_ns=period_ns):
                return self.check_error()
            await asyncio.sleep(0)

    def get_future(self):
        cdef PyObject* future_ptr

        with nogil:
            future_ptr = self._request.get().getPyFuture()

        return <object>future_ptr

    async def wait(self):
        if self._enable_python_future:
            await self.get_future()
        else:
            await self.wait_yield()


cdef class UCXBufferRequest:
    cdef:
        BufferRequestPtr _buffer_request
        bint _enable_python_future

    def __init__(self, uintptr_t shared_ptr_buffer_request, bint enable_python_future):
        self._buffer_request = deref(<BufferRequestPtr *> shared_ptr_buffer_request)
        self._enable_python_future = enable_python_future

    def get_request(self):
        return UCXRequest(
            <uintptr_t><void*>&self._buffer_request.get().request,
            self._enable_python_future,
        )

    def get_py_buffer(self):
        cdef Buffer* buf

        with nogil:
            buf = self._buffer_request.get().buffer

        # If buf == NULL, it holds a header
        if buf == NULL:
            return None
        elif buf.getType() == BufferType.RMM:
            return _get_rmm_buffer(<uintptr_t><void*>buf)
        else:
            return _get_host_buffer(<uintptr_t><void*>buf)


cdef class UCXBufferRequests:
    cdef:
        RequestTagMultiPtr _ucxx_request_tag_multi
        bint _enable_python_future
        bint _is_completed
        tuple _buffer_requests
        tuple _requests

    def __init__(self, uintptr_t unique_ptr_buffer_requests, bint enable_python_future):
        cdef RequestTagMulti ucxx_buffer_requests
        self._enable_python_future = enable_python_future
        self._is_completed = False
        self._requests = tuple()

        self._ucxx_request_tag_multi = (
            deref(<RequestTagMultiPtr *> unique_ptr_buffer_requests)
        )

    def _populate_requests(self):
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

            self._requests = tuple([br.get_request() for br in self._buffer_requests])

    def is_completed_all(self, int64_t period_ns=0):
        if self._is_completed is False:
            if self._ucxx_request_tag_multi.get()._isFilled is False:
                return False

            self._populate_requests()

            self._is_completed = all(
                [r.is_completed(period_ns=period_ns) for r in self._requests]
            )

        return self._is_completed

    def is_completed(self, int64_t period_ns=0):
        cdef bint is_completed

        if self._is_completed is False:
            with nogil:
                is_completed = self._ucxx_request_tag_multi.get().isCompleted(period_ns)
            self._is_completed = is_completed

        return self._is_completed

    async def wait_yield(self, period_ns=0):
        while True:
            if self.is_completed():
                for r in self._requests:
                    r.check_error()
                return
            await asyncio.sleep(0)

    async def _generate_future(self):
        if self._is_completed is False:
            while self._ucxx_request_tag_multi.get()._isFilled is False:
                await asyncio.sleep(0)

            self._populate_requests()

            futures = [r.get_future() for r in self._requests]
            await asyncio.gather(*futures)
            self._is_completed = True

        return self._is_completed

    def get_generator_future(self):
        return self._generate_future()

    def get_future(self):
        cdef PyObject* future_ptr

        with nogil:
            future_ptr = self._ucxx_request_tag_multi.get().getPyFuture()

        return <object>future_ptr

    async def wait(self):
        if self._enable_python_future:
            await self.get_future()
        else:
            await self.wait_yield()

    def get_requests(self):
        return self._requests

    def get_py_buffers(self):
        if not self.is_completed():
            raise RuntimeError("Some requests are not completed yet")

        self._populate_requests()

        py_buffers = [br.get_py_buffer() for br in self._buffer_requests]
        # PyBuffers that are None are headers
        return [b for b in py_buffers if b is not None]


cdef void _endpoint_close_callback(void *args) with gil:
    """Callback function called when UCXEndpoint closes or errors"""
    cdef dict cb_data = <dict> args

    try:
        cb_data['cb_func'](
            *cb_data['cb_args'],
            **cb_data['cb_kwargs']
        )
    except Exception as e:
        logger.debug(f"{type(e)} when calling endpoint close callback: {e}")


cdef class UCXEndpoint():
    cdef:
        shared_ptr[Endpoint] _endpoint
        uint64_t _context_feature_flags
        bint _enable_python_future
        dict _close_cb_data

    def __init__(
            self,
            uintptr_t shared_ptr_endpoint,
            bint enable_python_future,
            uint64_t context_feature_flags
    ):
        self._endpoint = deref(<shared_ptr[Endpoint] *> shared_ptr_endpoint)
        self._enable_python_future = enable_python_future
        self._context_feature_flags = context_feature_flags

    @classmethod
    def create(
            cls,
            UCXWorker worker,
            str ip_address,
            uint16_t port,
            bint endpoint_error_handling
    ):
        cdef shared_ptr[Context] context
        cdef shared_ptr[Endpoint] endpoint
        cdef string addr = ip_address.encode("utf-8")
        cdef uint64_t context_feature_flags

        with nogil:
            endpoint = worker._worker.get().createEndpointFromHostname(
                addr, port, endpoint_error_handling
            )
            context = dynamic_pointer_cast[Context, Component](
                worker._worker.get().getParent()
            )
            context_feature_flags = context.get().getFeatureFlags()

        return cls(
            <uintptr_t><void*>&endpoint,
            worker.is_python_future_enabled(),
            context_feature_flags
        )

    @classmethod
    def create_from_conn_request(
            cls,
            UCXListener listener,
            uintptr_t conn_request,
            bint endpoint_error_handling
    ):
        cdef shared_ptr[Context] context
        cdef shared_ptr[Worker] worker
        cdef shared_ptr[Endpoint] endpoint
        cdef uint64_t context_feature_flags

        with nogil:
            endpoint = listener._listener.get().createEndpointFromConnRequest(
                <ucp_conn_request_h>conn_request, endpoint_error_handling
            )
            worker = dynamic_pointer_cast[Worker, Component](
                listener._listener.get().getParent()
            )
            context = dynamic_pointer_cast[Context, Component](worker.get().getParent())
            context_feature_flags = context.get().getFeatureFlags()

        return cls(
            <uintptr_t><void*>&endpoint,
            listener.is_python_future_enabled(),
            context_feature_flags
        )

    @classmethod
    def create_from_worker_address(
            cls,
            UCXWorker worker,
            UCXAddress address,
            bint endpoint_error_handling
    ):
        cdef shared_ptr[Context] context
        cdef shared_ptr[Endpoint] endpoint
        cdef shared_ptr[Address] ucxx_address = address._address
        cdef uint64_t context_feature_flags

        with nogil:
            endpoint = worker._worker.get().createEndpointFromWorkerAddress(
                ucxx_address, endpoint_error_handling
            )
            context = dynamic_pointer_cast[Context, Component](
                worker._worker.get().getParent()
            )
            context_feature_flags = context.get().getFeatureFlags()

        return cls(
            <uintptr_t><void*>&endpoint,
            worker.is_python_future_enabled(),
            context_feature_flags
        )

    @property
    def handle(self):
        cdef ucp_ep_h handle

        with nogil:
            handle = self._endpoint.get().getHandle()

        return int(<uintptr_t>handle)

    def stream_send(self, Array arr):
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req

        if not self._context_feature_flags & Feature.STREAM.value:
            raise ValueError("UCXContext must be created with `Feature.STREAM`")

        with nogil:
            req = self._endpoint.get().streamSend(
                buf,
                nbytes,
                self._enable_python_future
            )

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def stream_recv(self, Array arr):
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req

        if not self._context_feature_flags & Feature.STREAM.value:
            raise ValueError("UCXContext must be created with `Feature.STREAM`")

        with nogil:
            req = self._endpoint.get().streamRecv(
                buf,
                nbytes,
                self._enable_python_future
            )

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def tag_send(self, Array arr, size_t tag):
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req

        if not self._context_feature_flags & Feature.TAG.value:
            raise ValueError("UCXContext must be created with `Feature.TAG`")

        with nogil:
            req = self._endpoint.get().tagSend(
                buf,
                nbytes,
                tag,
                self._enable_python_future
            )

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def tag_recv(self, Array arr, size_t tag):
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req

        if not self._context_feature_flags & Feature.TAG.value:
            raise ValueError("UCXContext must be created with `Feature.TAG`")

        with nogil:
            req = self._endpoint.get().tagRecv(
                buf,
                nbytes,
                tag,
                self._enable_python_future
            )

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def tag_send_multi(self, tuple arrays, size_t tag):
        cdef vector[void*] v_buffer
        cdef vector[size_t] v_size
        cdef vector[int] v_is_cuda
        cdef RequestTagMultiPtr ucxx_buffer_requests

        for arr in arrays:
            if not isinstance(arr, Array):
                raise ValueError(
                    "All elements of the `arrays` should be of `Array` type"
                )

        for arr in arrays:
            v_buffer.push_back(<void*><uintptr_t>arr.ptr)
            v_size.push_back(arr.nbytes)
            v_is_cuda.push_back(arr.cuda)

        with nogil:
            ucxx_buffer_requests = tagMultiSend(
                self._endpoint,
                v_buffer,
                v_size,
                v_is_cuda,
                tag,
                self._enable_python_future,
            )

        return UCXBufferRequests(
            <uintptr_t><void*>&ucxx_buffer_requests, self._enable_python_future,
        )

    def tag_send_multi_b(self, tuple buffer, tuple size, tuple is_cuda, size_t tag):
        cdef vector[void*] v_buffer
        cdef vector[size_t] v_size
        cdef vector[int] v_is_cuda

        for b, s, c in zip(buffer, size, is_cuda):
            v_buffer.push_back(<void*>b)
            v_size.push_back(s)
            v_is_cuda.push_back(c)

        with nogil:
            tagMultiSendBlocking(
                self._endpoint,
                v_buffer,
                v_size,
                v_is_cuda,
                tag,
                self._enable_python_future,
            )

    def tag_recv_multi(self, size_t tag):
        cdef RequestTagMultiPtr ucxx_buffer_requests

        with nogil:
            ucxx_buffer_requests = tagMultiRecv(
                self._endpoint, tag, self._enable_python_future
            )

        return UCXBufferRequests(
            <uintptr_t><void*>&ucxx_buffer_requests, self._enable_python_future,
        )

    def is_alive(self):
        cdef bint is_alive

        with nogil:
            is_alive = self._endpoint.get().isAlive()

        return is_alive

    def raise_on_error(self):
        with nogil:
            self._endpoint.get().raiseOnError()

    def set_close_callback(self, cb_func, tuple cb_args=None, dict cb_kwargs=None):
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}

        self._close_cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }

        cdef function[void(void*)]* func_close_callback = (
            new function[void(void*)](_endpoint_close_callback)
        )
        with nogil:
            self._endpoint.get().setCloseCallback(
                deref(func_close_callback), <void*>self._close_cb_data
            )
        del func_close_callback


cdef void _listener_callback(ucp_conn_request_h conn_request, void *args) with gil:
    """Callback function used by UCXListener"""
    cdef dict cb_data = <dict> args

    cb_data['cb_func'](
        (
            cb_data['listener'].create_endpoint_from_conn_request(
                int(<uintptr_t>conn_request), True
            ) if 'listener' in cb_data else
            int(<uintptr_t>conn_request)
        ),
        *cb_data['cb_args'],
        **cb_data['cb_kwargs']
    )


cdef class UCXListener():
    cdef:
        shared_ptr[Listener] _listener
        bint _enable_python_future
        dict _cb_data

    def __init__(
            self,
            uintptr_t shared_ptr_listener,
            dict cb_data,
            bint enable_python_future,
    ):
        self._listener = deref(<shared_ptr[Listener] *> shared_ptr_listener)
        self._cb_data = cb_data
        self._enable_python_future = enable_python_future

    @classmethod
    def create(
            cls,
            UCXWorker worker,
            uint16_t port,
            cb_func,
            tuple cb_args=None,
            dict cb_kwargs=None,
            bint deliver_endpoint=False,
    ):
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}

        cdef shared_ptr[Listener] ucxx_listener
        cdef ucp_listener_conn_callback_t listener_cb = (
            <ucp_listener_conn_callback_t>_listener_callback
        )
        cdef dict cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }

        with nogil:
            ucxx_listener = worker._worker.get().createListener(
                port, listener_cb, <void*>cb_data
            )

        listener = cls(
            <uintptr_t><void*>&ucxx_listener,
            cb_data,
            worker.is_python_future_enabled(),
        )
        if deliver_endpoint is True:
            cb_data["listener"] = listener
        return listener

    @property
    def port(self):
        cdef uint16_t port

        with nogil:
            port = self._listener.get().getPort()

        return port

    def create_endpoint_from_conn_request(
            self,
            uintptr_t conn_request,
            bint endpoint_error_handling
    ):
        return UCXEndpoint.create_from_conn_request(
            self, conn_request, endpoint_error_handling,
        )

    def is_python_future_enabled(self):
        return self._enable_python_future


def get_ucx_version():
    cdef unsigned int a, b, c
    ucp_get_version(&a, &b, &c)
    return (a, b, c)
