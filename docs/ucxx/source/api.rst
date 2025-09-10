API
===

.. currentmodule:: ucxx

**ucxx**

.. autosummary::
   ucxx
   ucxx.create_listener
   ucxx.create_endpoint
   ucxx.get_address
   ucxx.get_config
   ucxx.get_ucp_worker
   ucxx.get_ucx_version
   ucxx.init
   ucxx.progress
   ucxx.reset

**Endpoint**

.. autosummary::
   Endpoint
   Endpoint.abort
   Endpoint.close
   Endpoint.closed
   Endpoint.close_after_n_recv
   Endpoint.cuda_support
   Endpoint.get_ucp_endpoint
   Endpoint.get_ucp_worker
   Endpoint.recv
   Endpoint.send
   Endpoint.ucx_info
   Endpoint.uid

**Listener**

.. autosummary::
   Listener
   Listener.close
   Listener.closed
   Listener.port

.. currentmodule:: ucxx

.. autofunction:: create_listener
.. autofunction:: create_endpoint
.. autofunction:: get_address
.. autofunction:: get_config
.. autofunction:: get_ucp_worker
.. autofunction:: get_ucx_version
.. autofunction:: init
.. autofunction:: progress
.. autofunction:: reset

Endpoint
--------

.. currentmodule:: ucxx

.. autoclass:: Endpoint
   :members:


Listener
--------

.. currentmodule:: ucxx

.. autoclass:: Listener
   :members:
