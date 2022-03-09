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


class UCXError(Exception):
    pass


class UCXCanceled(UCXError):
    pass


class UCXConfigError(UCXError):
    pass


class UCXConnectionResetError(UCXError):
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
        cdef cpp_map[string, string] cpp_config
        for k, v in config_dict.items():
            cpp_config[k.encode("utf-8")] = v.encode("utf-8")
        feature_flags = functools.reduce(
            lambda x, y: x | y.value, feature_flags, 0
        )
        self._context = UCXXContext.create(cpp_config, feature_flags)
        cdef UCXXContext* ucxx_context = self._context.get()

        cdef dict context_config = ucxx_context.get_config()
        self._config = {
            k.decode("utf-8"): v.decode("utf-8") for k, v in context_config.items()
        }

    cpdef dict get_config(self):
        return self._config

    @property
    def handle(self):
        cdef UCXXContext* ucxx_context = self._context.get()
        return int(<uintptr_t>ucxx_context.get_handle())

    cpdef str get_info(self):
        cdef UCXXContext* ucxx_context = self._context.get()
        return ucxx_context.get_info().decode("utf-8")


cdef class UCXAddress():
    cdef:
        shared_ptr[UCXXAddress] _address

    def __init__(self, uintptr_t shared_ptr_address):
        self._address = (<shared_ptr[UCXXAddress] *> shared_ptr_address)[0]

    @classmethod
    def create_from_worker(cls, UCXWorker worker):
        cdef UCXXWorker* ucxx_worker = worker._worker.get()
        address = ucxx_worker.getAddress()
        return cls(<uintptr_t><void*>&address)

    # For old UCX-Py API compatibility
    @classmethod
    def from_worker(cls, UCXWorker worker):
        return cls.create_from_worker(worker)

    @property
    def address(self):
        cdef UCXXAddress* ucxx_address = self._address.get()
        return int(<uintptr_t>ucxx_address.getHandle())

    @property
    def length(self):
        cdef UCXXAddress* ucxx_address = self._address.get()
        return int(ucxx_address.getLength())


cdef class UCXWorker():
    """Python representation of `ucp_worker_h`"""
    cdef:
        shared_ptr[UCXXWorker] _worker

    def __init__(self, UCXContext context):
        cdef UCXXContext* ucxx_context = context._context.get()
        self._worker = ucxx_context.createWorker()

    @property
    def handle(self):
        cdef UCXXWorker* ucxx_worker = self._worker.get()
        return int(<uintptr_t>ucxx_worker.get_handle())

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
        cdef UCXXWorker* ucxx_worker = self._worker.get()
        ucxx_worker.init_blocking_progress_mode()

    def progress(self):
        cdef UCXXWorker* ucxx_worker = self._worker.get()
        ucxx_worker.progress()

    def progress_worker_event(self):
        cdef UCXXWorker* ucxx_worker = self._worker.get()
        ucxx_worker.progress_worker_event()

    def start_progress_thread(self):
        cdef UCXXWorker* ucxx_worker = self._worker.get()
        ucxx_worker.startProgressThread()

    def stop_progress_thread(self):
        cdef UCXXWorker* ucxx_worker = self._worker.get()
        ucxx_worker.stopProgressThread()

    def cancel_inflight_requests(self):
        cdef UCXXWorker* ucxx_worker = self._worker.get()
        return ucxx_worker.cancelInflightRequests()

    def tag_probe(self, tag):
        cdef UCXXWorker* ucxx_worker = self._worker.get()
        return ucxx_worker.tagProbe(tag)


cdef class UCXRequest():
    cdef:
        shared_ptr[UCXXRequest] _request

    def __init__(self, uintptr_t shared_ptr_request):
        self._request = (<shared_ptr[UCXXRequest] *> shared_ptr_request)[0]

    def is_completed(self, period_ns=0):
        cdef UCXXRequest* ucxx_request = self._request.get()
        return ucxx_request.isCompleted(period_ns)

    def get_status(self):
        cdef UCXXRequest* ucxx_request = self._request.get()
        return ucxx_request.getStatus()

    def check_error(self):
        cdef UCXXRequest* ucxx_request = self._request.get()
        return ucxx_request.checkError()

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
        cdef UCXXWorker* ucxx_worker = worker._worker.get()
        endpoint = ucxx_worker.createEndpointFromHostname(
            ip_address.encode("utf-8"), port, endpoint_error_handling
        )
        return cls(<uintptr_t><void*>&endpoint)

    @classmethod
    def create_from_conn_request(
            cls,
            UCXListener listener,
            uintptr_t conn_request,
            bint endpoint_error_handling
    ):
        cdef UCXXListener* ucxx_listener = listener._listener.get()
        endpoint = ucxx_listener.createEndpointFromConnRequest(
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
        cdef UCXXWorker* ucxx_worker = worker._worker.get()
        cdef shared_ptr[UCXXAddress] ucxx_address = address._address
        endpoint = ucxx_worker.createEndpointFromWorkerAddress(
            ucxx_address, endpoint_error_handling
        )
        return cls(<uintptr_t><void*>&endpoint)

    @property
    def handle(self):
        cdef UCXXEndpoint* e = self._endpoint.get()
        return int(<uintptr_t>e.getHandle())

    def stream_send(self, Array arr):
        cdef UCXXEndpoint* ucxx_endpoint = self._endpoint.get()
        cdef shared_ptr[UCXXRequest] req = (
            ucxx_endpoint.stream_send(<void*>arr.ptr, arr.nbytes)
        )
        return UCXRequest(<uintptr_t><void*>&req)

    def stream_recv(self, Array arr):
        cdef UCXXEndpoint* ucxx_endpoint = self._endpoint.get()
        cdef shared_ptr[UCXXRequest] req = (
            ucxx_endpoint.stream_recv(<void*>arr.ptr, arr.nbytes)
        )
        return UCXRequest(<uintptr_t><void*>&req)

    def tag_send(self, Array arr, int tag):
        cdef UCXXEndpoint* ucxx_endpoint = self._endpoint.get()
        cdef shared_ptr[UCXXRequest] req = (
            ucxx_endpoint.tag_send(<void*>arr.ptr, arr.nbytes, tag)
        )
        return UCXRequest(<uintptr_t><void*>&req)

    def tag_recv(self, Array arr, int tag):
        cdef UCXXEndpoint* ucxx_endpoint = self._endpoint.get()
        cdef shared_ptr[UCXXRequest] req = (
            ucxx_endpoint.tag_recv(<void*>arr.ptr, arr.nbytes, tag)
        )
        return UCXRequest(<uintptr_t><void*>&req)

    def is_alive(self):
        cdef UCXXEndpoint* ucxx_endpoint = self._endpoint.get()
        return ucxx_endpoint.isAlive()

    def raise_on_error(self):
        cdef UCXXEndpoint* ucxx_endpoint = self._endpoint.get()
        return ucxx_endpoint.raiseOnError()

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

        cdef UCXXEndpoint* ucxx_endpoint = self._endpoint.get()

        cdef function[void(void*)]* func_close_callback = (
            new function[void(void*)](_endpoint_close_callback)
        )
        ucxx_endpoint.setCloseCallback(
            func_close_callback[0], <void*>self._close_cb_data
        )
        del func_close_callback


cdef void _listener_callback(ucp_conn_request_h conn_request, void *args) with gil:
    """Callback function used by UCXListener"""
    cdef dict cb_data = <dict> args

    cb_data['cb_func'](
        int(<uintptr_t>conn_request),
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
            dict cb_kwargs=None
    ):
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}

        cdef UCXXWorker* ucxx_worker = worker._worker.get()
        cdef ucp_listener_conn_callback_t listener_cb = (
            <ucp_listener_conn_callback_t>_listener_callback
        )
        cdef dict cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }

        listener = ucxx_worker.createListener(port, listener_cb, <void*>cb_data)
        return cls(<uintptr_t><void*>&listener, cb_data)

    @property
    def port(self):
        cdef UCXXListener* ucxx_listener = self._listener.get()
        return ucxx_listener.getPort()

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
