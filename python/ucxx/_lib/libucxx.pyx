# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# distutils: language = c++
# cython: language_level=3

import enum
import functools

from libc.stdint cimport uintptr_t
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from . cimport ucxx_api
from .ucxx_api cimport *


class Feature(enum.Enum):
    TAG = UCP_FEATURE_TAG
    RMA = UCP_FEATURE_RMA
    AMO32 = UCP_FEATURE_AMO32
    AMO64 = UCP_FEATURE_AMO64
    WAKEUP = UCP_FEATURE_WAKEUP
    STREAM = UCP_FEATURE_STREAM
    AM = UCP_FEATURE_AM


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
        cdef UCXXContext* context = self._context.get()

        cdef dict context_config = context.get_config()
        self._config = {k.decode("utf-8"): v.decode("utf-8") for k, v in context_config.items()}

    cpdef dict get_config(self):
        return self._config

    @property
    def handle(self):
        cdef UCXXContext* context = self._context.get()
        return int(<uintptr_t>context.get_handle())

    cpdef str get_info(self):
        cdef UCXXContext* context = self._context.get()
        return context.get_info().decode("utf-8")


cdef class UCXAddress():
    cdef:
        shared_ptr[UCXXAddress] _address

    def __init__(self, uintptr_t shared_ptr_address):
        self._address = (<shared_ptr[UCXXAddress] *> shared_ptr_address)[0]

    @classmethod
    def create_from_worker(cls, UCXWorker worker):
        cdef UCXXWorker* w = worker._worker.get()
        address = w.getAddress()
        return cls(<uintptr_t><void*>&address)

    # For old UCX-Py API compatibility
    @classmethod
    def from_worker(cls, UCXWorker worker):
        return cls.create_from_worker(worker)

    @property
    def address(self):
        cdef shared_ptr[UCXXAddress] address = self._address
        return <uintptr_t>address.get_handle()

    @property
    def length(self):
        cdef shared_ptr[UCXXAddress] address = self._address
        return int(address.getLength())


cdef class UCXWorker():
    """Python representation of `ucp_worker_h`"""
    cdef:
        shared_ptr[UCXXWorker] _worker

    def __init__(self, UCXContext context):
        cdef UCXXContext* ctx = context._context.get()
        self._worker = ctx.createWorker()

    @property
    def handle(self):
        cdef UCXXWorker* worker = self._worker.get()
        return int(<uintptr_t>worker.get_handle())

    def get_address(self):
        return UCXAddress.create_from_worker(self)

    def createEndpointFromHostname(self, str ip_address, uint16_t port, bint endpoint_error_handling):
        return UCXEndpoint.create(self, ip_address, port, endpoint_error_handling)

    def createEndpointFromWorkerAddress(self, UCXAddress address, bint endpoint_error_handling):
        return UCXEndpoint.create_from_worker_address(self, address, endpoint_error_handling)

    def progress(self):
        cdef UCXXWorker* worker = self._worker.get()
        worker.progress()


cdef class UCXEndpoint():
    cdef:
        shared_ptr[UCXXEndpoint] _endpoint

    def __init__(self, uintptr_t shared_ptr_endpoint):
        self._endpoint = (<shared_ptr[UCXXEndpoint] *> shared_ptr_endpoint)[0]

    @classmethod
    def create(cls, UCXWorker worker, str ip_address, uint16_t port, bint endpoint_error_handling):
        cdef UCXXWorker* w = worker._worker.get()
        endpoint = w.createEndpointFromHostname(ip_address.encode("utf-8"), port, endpoint_error_handling)
        return cls(<uintptr_t><void*>&endpoint)

    @classmethod
    def create_from_conn_request(cls, UCXListener listener, uintptr_t conn_request, bint endpoint_error_handling):
        cdef UCXXListener* l = listener._listener.get()
        endpoint = l.createEndpointFromConnRequest(<ucp_conn_request_h>conn_request, endpoint_error_handling)
        return cls(<uintptr_t><void*>&endpoint)

    @classmethod
    def create_from_worker_address(cls, UCXWorker worker, UCXAddress address, bint endpoint_error_handling):
        cdef UCXXWorker* w = worker._worker.get()
        cdef shared_ptr[UCXXAddress] a = address._address
        endpoint = w.createEndpointFromWorkerAddress(a, endpoint_error_handling)
        return cls(<uintptr_t><void*>&endpoint)


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
    def create(cls, UCXWorker worker, uint16_t port, cb_func, tuple cb_args=None, dict cb_kwargs=None):
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}

        cdef UCXXWorker* w = worker._worker.get()
        cdef ucp_listener_conn_callback_t listener_cb = (
            <ucp_listener_conn_callback_t>_listener_callback
        )
        cdef dict cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }

        listener = w.createListener(port, listener_cb, <void*>cb_data)
        return cls(<uintptr_t><void*>&listener, cb_data)

    def createEndpointFromConnRequest(self, uintptr_t conn_request, bint endpoint_error_handling):
        return UCXEndpoint.create_from_conn_request(self, conn_request, endpoint_error_handling)
