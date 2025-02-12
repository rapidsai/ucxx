from libc.stdint cimport uint64_t
from libcpp.memory cimport shared_ptr, unique_ptr

from .ucxx_api cimport *


cdef class HostBufferAdapter:
    cdef Py_ssize_t _size
    cdef void* _ptr
    cdef Py_ssize_t[1] _shape
    cdef Py_ssize_t[1] _strides
    cdef Py_ssize_t _itemsize


cdef class UCXConfig:
    cdef:
        unique_ptr[Config] _config
        bint _enable_python_future
        dict _cb_data


cdef class UCXContext:
    cdef:
        shared_ptr[Context] _context
        dict _config


cdef class UCXAddress:
    cdef:
        shared_ptr[Address] _address
        size_t _length
        ucp_address_t *_handle
        string _string


cdef class UCXWorker:
    cdef:
        shared_ptr[Worker] _worker
        dict _progress_thread_start_cb_data
        bint _enable_delayed_submission
        bint _enable_python_future
        uint64_t _context_feature_flags


cdef class UCXRequest:
    cdef:
        shared_ptr[Request] _request
        bint _enable_python_future
        bint _completed


cdef class UCXBufferRequest:
    cdef:
        BufferRequestPtr _buffer_request
        bint _enable_python_future


cdef class UCXBufferRequests:
    cdef:
        RequestTagMultiPtr _ucxx_request_tag_multi
        bint _enable_python_future
        bint _completed
        tuple _buffer_requests
        tuple _requests


cdef class UCXEndpoint:
    cdef:
        shared_ptr[Endpoint] _endpoint
        uint64_t _context_feature_flags
        bint _cuda_support
        bint _enable_python_future
        dict _close_cb_data
        shared_ptr[uintptr_t] _close_cb_data_ptr


cdef class UCXListener:
    cdef:
        shared_ptr[Listener] _listener
        bint _enable_python_future
        dict _cb_data
        object __weakref__
