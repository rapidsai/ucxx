# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

import asyncio
import enum
import functools

from cpython.ref cimport PyObject
from libc.stdint cimport uintptr_t
from libcpp.functional cimport function
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from . cimport ucxx_api
from .arr cimport Array
from .ucxx_api cimport *

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

    def start_progress_thread(self):
        with nogil:
            self._worker.get().startProgressThread()

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

    def __init__(self, uintptr_t shared_ptr_request):
        self._request = (<shared_ptr[UCXXRequest] *> shared_ptr_request)[0]

    def is_completed(self, int64_t period_ns=0):
        cdef bint is_completed

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

    async def is_completed_async(self, period_ns=0):
        while True:
            if self.is_completed():
                return self.check_error()
            await asyncio.sleep(0)


cdef void _endpoint_close_callback(void *args) with gil:
    """Callback function called when UCXEndpoint closes or errors"""
    cdef dict cb_data = <dict> args

    cb_data['cb_func'](
        *cb_data['cb_args'],
        **cb_data['cb_kwargs']
    )


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
