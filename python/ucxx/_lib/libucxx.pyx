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
from libc.stdint cimport uintptr_t
from libcpp.cast cimport dynamic_cast
from libcpp.functional cimport function
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from . cimport ucxx_api
from .arr cimport Array
from .ucxx_api cimport *

from rmm._lib.device_buffer cimport DeviceBuffer

logger = logging.getLogger("ucx")


import numpy as np
np.import_array()


cdef ptr_to_ndarray(void* ptr, np.npy_intp N):
    cdef np.ndarray[np.uint8_t, ndim=1, mode="c"] arr = (
        np.PyArray_SimpleNewFromData(1, &N, np.NPY_UINT8, <np.uint8_t*>ptr)
    )
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA)
    return arr


def _get_rmm_buffer(uintptr_t unique_ptr_recv_buffer):
    cdef unique_ptr[UCXXPyBuffer] recv_buffer = move((<unique_ptr[UCXXPyBuffer]*> unique_ptr_recv_buffer)[0])
    cdef UCXXPyRMMBufferPtr rmm_buffer = dynamic_cast[UCXXPyRMMBufferPtr](recv_buffer.get())
    return DeviceBuffer.c_from_unique_ptr(move(rmm_buffer.get()))


def _get_host_buffer(uintptr_t unique_ptr_recv_buffer):
    cdef unique_ptr[UCXXPyBuffer] recv_buffer = move((<unique_ptr[UCXXPyBuffer]*> unique_ptr_recv_buffer)[0])
    cdef UCXXPyHostBufferPtr host_buffer = dynamic_cast[UCXXPyHostBufferPtr](recv_buffer.get())
    return ptr_to_ndarray(host_buffer.release(), host_buffer.getSize())


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


###############################################################################
#                                   Classes                                   #
###############################################################################

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
        shared_ptr[UCXXContext] _context
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
        cdef cpp_map[string, string] cpp_config_in, cpp_config_out
        cdef dict context_config

        for k, v in config_dict.items():
            cpp_config_in[k.encode("utf-8")] = v.encode("utf-8")
        cdef uint64_t feature_flags_uint = functools.reduce(
            lambda x, y: x | y.value, feature_flags, 0
        )

        with nogil:
            self._context = UCXXContext.create(cpp_config_in, feature_flags_uint)
            cpp_config_out = self._context.get().get_config()

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
            handle = self._context.get().get_handle()

        return int(<uintptr_t>handle)

    cpdef str get_info(self):
        cdef UCXXContext* ucxx_context
        cdef string info

        with nogil:
            ucxx_context = self._context.get()
            info = ucxx_context.get_info()

        return info.decode("utf-8")


cdef class UCXAddress():
    cdef:
        shared_ptr[UCXXAddress] _address

    def __init__(self, uintptr_t shared_ptr_address):
        self._address = (<shared_ptr[UCXXAddress] *> shared_ptr_address)[0]

    @classmethod
    def create_from_worker(cls, UCXWorker worker):
        cdef shared_ptr[UCXXAddress] address

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
        shared_ptr[UCXXWorker] _worker
        dict _progress_thread_start_cb_data

    def __init__(self, UCXContext context):
        with nogil:
            self._worker = context._context.get().createWorker()

    @property
    def handle(self):
        cdef ucp_worker_h handle

        with nogil:
            handle = self._worker.get().get_handle()

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
            self._worker.get().init_blocking_progress_mode()

    def progress(self):
        with nogil:
            self._worker.get().progress()

    def progress_once(self):
        cdef bint progress_made

        with nogil:
            progress_made = self._worker.get().progress_once()

        return progress_made

    def progress_worker_event(self):
        with nogil:
            self._worker.get().progress_worker_event()

    def start_progress_thread(self, bint polling_mode=True):
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

    def set_progress_thread_start_callback(self, cb_func, tuple cb_args=None, dict cb_kwargs=None):
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
                func_generic_callback[0], <void*>self._progress_thread_start_cb_data
            )
        del func_generic_callback


cdef class UCXRequest():
    cdef:
        shared_ptr[UCXXRequest] _request
        bint _is_completed

    def __init__(self, uintptr_t shared_ptr_request):
        self._request = (<shared_ptr[UCXXRequest] *> shared_ptr_request)[0]
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
        cdef PyObject* future_ptr = NULL

        with nogil:
            future_ptr = self._request.get().getPyFuture()

        return <object>future_ptr



cdef class UCXBufferRequest:
    cdef:
        UCXXBufferRequestPtr _buffer_request

    def __init__(self, uintptr_t shared_ptr_buffer_request):
        self._buffer_request = (<UCXXBufferRequestPtr *> shared_ptr_buffer_request)[0]

    def get_request(self):
        return UCXRequest(<uintptr_t><void*>&self._buffer_request.get().request)

    def get_py_buffer(self):
        cdef unique_ptr[UCXXPyBuffer] buf

        with nogil:
            buf = move(self._buffer_request.get().pyBuffer)

        # If pyBuffer == NULL, it holds a header
        if buf.get() == NULL:
            return None
        elif buf.get().isCUDA():
            return _get_rmm_buffer(<uintptr_t><void*>&buf)
        else:
            return _get_host_buffer(<uintptr_t><void*>&buf)


cdef class UCXBufferRequests:
    cdef:
        UCXXBufferRequestsPtr _ucxx_buffer_requests
        bint _is_completed
        tuple _buffer_requests
        tuple _requests

    def __init__(self, uintptr_t unique_ptr_buffer_requests):
        cdef UCXXBufferRequests ucxx_buffer_requests
        self._is_completed = False
        self._requests = tuple()

        self._ucxx_buffer_requests = (<UCXXBufferRequestsPtr *> unique_ptr_buffer_requests)[0]

    def _populate_requests(self):
        if len(self._requests) == 0:
            ucxx_buffer_requests = self._ucxx_buffer_requests.get()[0]
            self._buffer_requests = tuple([
                UCXBufferRequest(<uintptr_t><void*>&(ucxx_buffer_requests.bufferRequests[i]))
                for i in range(ucxx_buffer_requests.bufferRequests.size())
            ])

            self._requests = tuple([br.get_request() for br in self._buffer_requests])

    def is_completed(self, int64_t period_ns=0):
        if self._is_completed is False:
            if self._ucxx_buffer_requests.get().isFilled is False:
                return False

            self._populate_requests()

            self._is_completed = all([r.is_completed(period_ns=period_ns) for r in self._requests])

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
            while self._ucxx_buffer_requests.get().isFilled is False:
                await asyncio.sleep(0)

            self._populate_requests()

            futures = [r.get_future() for r in self._requests]
            await asyncio.gather(*futures)
            self._is_completed = True

        return self._is_completed

    def get_future(self):
        return self._generate_future()

    def get_requests(self):
        return self._requests

    def get_py_buffers(self):
        if not self.is_completed():
            raise RuntimeError("Some requests are not completed yet")

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
        shared_ptr[UCXXEndpoint] _endpoint
        dict _close_cb_data

    def __init__(self, uintptr_t shared_ptr_endpoint):
        self._endpoint = (<shared_ptr[UCXXEndpoint] *> shared_ptr_endpoint)[0]

    @classmethod
    def create(
            cls,
            UCXWorker worker,
            str ip_address,
            uint16_t port,
            bint endpoint_error_handling
    ):
        cdef shared_ptr[UCXXEndpoint] endpoint
        cdef string addr = ip_address.encode("utf-8")

        with nogil:
            endpoint = worker._worker.get().createEndpointFromHostname(
                addr, port, endpoint_error_handling
            )

        return cls(<uintptr_t><void*>&endpoint)

    @classmethod
    def create_from_conn_request(
            cls,
            UCXListener listener,
            uintptr_t conn_request,
            bint endpoint_error_handling
    ):
        cdef shared_ptr[UCXXEndpoint] endpoint

        with nogil:
            endpoint = listener._listener.get().createEndpointFromConnRequest(
                <ucp_conn_request_h>conn_request, endpoint_error_handling
            )

        return cls(<uintptr_t><void*>&endpoint)

    @classmethod
    def create_from_worker_address(
            cls,
            UCXWorker worker,
            UCXAddress address,
            bint endpoint_error_handling
    ):
        cdef shared_ptr[UCXXAddress] ucxx_address = address._address
        cdef shared_ptr[UCXXEndpoint] endpoint

        with nogil:
            endpoint = worker._worker.get().createEndpointFromWorkerAddress(
                ucxx_address, endpoint_error_handling
            )

        return cls(<uintptr_t><void*>&endpoint)

    @property
    def handle(self):
        cdef ucp_ep_h handle

        with nogil:
            handle = self._endpoint.get().getHandle()

        return int(<uintptr_t>handle)

    def stream_send(self, Array arr):
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[UCXXRequest] req

        with nogil:
            req = self._endpoint.get().stream_send(buf, nbytes)

        return UCXRequest(<uintptr_t><void*>&req)

    def stream_recv(self, Array arr):
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[UCXXRequest] req

        with nogil:
            req = self._endpoint.get().stream_recv(buf, nbytes)

        return UCXRequest(<uintptr_t><void*>&req)

    def tag_send(self, Array arr, size_t tag):
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[UCXXRequest] req

        with nogil:
            req = self._endpoint.get().tag_send(buf, nbytes, tag)

        return UCXRequest(<uintptr_t><void*>&req)

    def tag_recv(self, Array arr, size_t tag):
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[UCXXRequest] req

        with nogil:
            req = self._endpoint.get().tag_recv(buf, nbytes, tag)

        return UCXRequest(<uintptr_t><void*>&req)

    # def tag_send_multi(self, tuple buffer, tuple size, tuple is_cuda, size_t tag):
    #     cdef vector[void*] v_buffer
    #     cdef vector[size_t] v_size
    #     cdef vector[int] v_is_cuda

    #     for b, s, c in zip(buffer, size, is_cuda):
    #         v_buffer.push_back(<void*><uintptr_t>b)
    #         v_size.push_back(s)
    #         v_is_cuda.push_back(c)

    #     cdef UCXXBufferRequestsPtr ucxx_buffer_requests

    #     with nogil:
    #         ucxx_buffer_requests = move(tag_send_multi(self._endpoint, v_buffer, v_size, v_is_cuda, tag))

    #     return UCXBufferRequests(<uintptr_t><void*>&ucxx_buffer_requests)

    def tag_send_multi(self, tuple arrays, size_t tag):
        cdef vector[void*] v_buffer
        cdef vector[size_t] v_size
        cdef vector[int] v_is_cuda
        cdef UCXXBufferRequestsPtr ucxx_buffer_requests

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
            ucxx_buffer_requests = tag_send_multi(self._endpoint, v_buffer, v_size, v_is_cuda, tag)

        return UCXBufferRequests(<uintptr_t><void*>&ucxx_buffer_requests)

    def tag_send_multi_b(self, tuple buffer, tuple size, tuple is_cuda, size_t tag):
        cdef vector[void*] v_buffer
        cdef vector[size_t] v_size
        cdef vector[int] v_is_cuda

        for b, s, c in zip(buffer, size, is_cuda):
            v_buffer.push_back(<void*><uintptr_t>b)
            v_size.push_back(s)
            v_is_cuda.push_back(c)

        with nogil:
            tag_send_multi_b(self._endpoint, v_buffer, v_size, v_is_cuda, tag)

    def tag_recv_multi(self, size_t tag):
        cdef UCXXBufferRequestsPtr ucxx_buffer_requests

        with nogil:
            buffer_requests = move(tag_recv_multi(self._endpoint, tag))

        return UCXBufferRequests(<uintptr_t><void*>&buffer_requests)

    # def tag_recv_multi_b(self, size_t tag):
    #     cdef vector[unique_ptr[UCXXPyBuffer]] recv_buffers
    #     cdef list buffers = []

    #     with nogil:
    #         recv_buffers = tag_recv_multi_b(self._endpoint, tag)

    #     for i in range(recv_buffers.size()):
    #         if recv_buffers[i].get().isCUDA():
    #             buffers.append(_get_rmm_buffer(<uintptr_t><void*>&recv_buffers[i]))
    #         else:
    #             buffers.append(_get_host_buffer(<uintptr_t><void*>&recv_buffers[i]))

    #     return buffers


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
                func_close_callback[0], <void*>self._close_cb_data
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
        shared_ptr[UCXXListener] _listener
        dict _cb_data

    def __init__(self, uintptr_t shared_ptr_listener, dict cb_data):
        self._listener = (<shared_ptr[UCXXListener] *> shared_ptr_listener)[0]
        self._cb_data = cb_data

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

        cdef shared_ptr[UCXXListener] ucxx_listener
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

        listener = cls(<uintptr_t><void*>&ucxx_listener, cb_data)
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


def get_ucx_version():
    cdef unsigned int a, b, c
    ucp_get_version(&a, &b, &c)
    return (a, b, c)
