from libc.stdint cimport uint64_t
from libcpp.memory cimport shared_ptr

from .ucxx_api cimport *


cdef class UCXAddress():
    cdef:
        shared_ptr[Address] _address
        size_t _length
        ucp_address_t *_handle
        string _string


cdef class UCXWorker():
    cdef:
        shared_ptr[Worker] _worker
        dict _progress_thread_start_cb_data
        bint _enable_delayed_submission
        bint _enable_python_future
        uint64_t _context_feature_flags
