# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.)
# SPDX-License-Identifier: BSD-3-Clause


import array
import asyncio
import logging
import warnings

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx._lib.libucxx import UCXCanceled, UCXCloseError, UCXError
from ucxx.types import Tag, TagMaskFull

from .utils import hash64bits

logger = logging.getLogger("ucx")


class Endpoint:
    """An endpoint represents a connection to a peer

    Please use `create_listener()` and `create_endpoint()`
    to create an Endpoint.
    """

    def __init__(self, endpoint, ctx, tags=None):
        from .application_context import ApplicationContext

        if not isinstance(endpoint, ucx_api.UCXEndpoint):
            raise ValueError("endpoint must be an instance of UCXEndpoint")
        if not isinstance(ctx, ApplicationContext):
            raise ValueError("ctx must be an instance of ApplicationContext")

        self._ep = endpoint
        self._ctx = ctx
        self._send_count = 0  # Number of calls to self.send()
        self._recv_count = 0  # Number of calls to self.recv()
        self._finished_recv_count = 0  # Number of returned (finished) self.recv() calls
        self._shutting_down_peer = False  # Told peer to shutdown
        self._close_after_n_recv = None
        self._tags = tags

    def __del__(self):
        self.abort()

    @property
    def alive(self):
        return self._ep.alive

    @property
    def closed(self):
        """Is this endpoint closed?"""
        return self._ep is None or not self.alive

    @property
    def ucp_endpoint(self):
        """The underlying UCP endpoint handle (ucp_ep_h) as a Python integer."""
        return self._ep.handle

    @property
    def ucp_worker(self):
        """The underlying UCP worker handle (ucp_worker_h) as a Python integer."""
        return self._ctx.worker.handle

    @property
    def ucxx_endpoint(self):
        """The underlying UCXX endpoint pointer (ucxx::Endpoint*) as a Python
        integer.
        """
        return self._ep.ucxx_ptr

    @property
    def ucxx_worker(self):
        """Returns the underlying UCXX worker pointer (ucxx::Worker*)
        as a Python integer.
        """
        return self._ctx.worker.ucxx_ptr

    @property
    def uid(self):
        """The unique ID of the underlying UCX endpoint"""
        return self._ep.handle

    def abort(self, period=10**10, max_attempts=1):
        """Close the communication immediately and abruptly.
        Useful in destructors or generators' ``finally`` blocks.

        Despite the attempt to close communication immediately, in some
        circumstances, notably when the parent worker is running a progress
        thread, a maximum timeout may be specified for which the close operation
        will wait. This can be particularly important for cases where the progress
        thread might be attempting to acquire the GIL while the current
        thread owns that resource.

        Notice, this functions doesn't signal the connected peer to close.
        To do that, use `Endpoint.close()`.

        Parameters
        ----------
        period: int
            maximum period to wait (in ns) for internal endpoint operations
            to complete, usually two operations (pre and post) are involved
            thus the maximum perceived timeout should be multiplied by two.
        max_attempts: int
            maximum number of attempts to close endpoint, only applicable
            if worker is running a progress thread and `period > 0`.
        """
        if self._ep is not None:
            logger.debug("Endpoint.abort(): 0x%x" % self.uid)
            # Wait for a maximum of `period` ns
            self._ep.close_blocking(period=period, max_attempts=max_attempts)
            self._ep.remove_close_callback()
        self._ep = None
        self._ctx = None

    async def close(self, period=10**10, max_attempts=1):
        """Close the endpoint cleanly.
        This will attempt to flush outgoing buffers before actually
        closing the underlying UCX endpoint.

        A maximum timeout and number of attempts may be specified to prevent the
        underlying `Endpoint` object from failing to acquire the GIL, see `abort()`
        for details.

        Parameters
        ----------
        period: int
            maximum period to wait (in ns) for internal endpoint operations
            to complete, usually two operations (pre and post) are involved
            thus the maximum perceived timeout should be multiplied by two.
        max_attempts: int
            maximum number of attempts to close endpoint, only applicable
            if worker is running a progress thread and `period > 0`.
        """
        if self.closed:
            self.abort(period=period, max_attempts=max_attempts)
            return
        try:
            # Making sure we only tell peer to shutdown once
            if self._shutting_down_peer:
                return
            self._shutting_down_peer = True

        finally:
            if not self.closed:
                # Give all current outstanding send() calls a chance to return
                if not self._ctx.progress_mode.startswith("thread"):
                    self._ctx.worker.progress()
                await asyncio.sleep(0)
                self.abort(period=period, max_attempts=max_attempts)

    async def am_send(self, buffer):
        """Send `buffer` to connected peer via active messages.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to send. Raise ValueError if buffer is smaller
            than nbytes.
        """
        self._ep.raise_on_error()
        if self.closed:
            raise UCXCloseError("Endpoint closed")
        if not isinstance(buffer, Array):
            buffer = Array(buffer)

        # Optimization to eliminate producing logger string overhead
        if logger.isEnabledFor(logging.DEBUG):
            nbytes = buffer.nbytes
            log = "[AM Send #%03d] ep: 0x%x, nbytes: %d, type: %s" % (
                self._send_count,
                self.uid,
                nbytes,
                type(buffer.obj),
            )
            logger.debug(log)

        self._send_count += 1

        try:
            request = self._ep.am_send(buffer)
            return await request.wait()
        except UCXCanceled as e:
            # If self._ep has already been closed and destroyed, we reraise the
            # UCXCanceled exception.
            if self._ep is None:
                raise e

    # @ucx_api.nvtx_annotate("UCXPY_SEND", color="green", domain="ucxpy")
    async def send(self, buffer, tag=None, force_tag=False):
        """Send `buffer` to connected peer.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to send. Raise ValueError if buffer is smaller
            than nbytes.
        tag: hashable, optional
            Set a tag that the receiver must match. Currently the tag
            is hashed together with the internal Endpoint tag that is
            agreed with the remote end at connection time. To enforce
            using the user tag, make sure to specify `force_tag=True`.
        force_tag: bool
            If true, force using `tag` as is, otherwise the value
            specified with `tag` (if any) will be hashed with the
            internal Endpoint tag.
        """
        self._ep.raise_on_error()
        if self.closed:
            raise UCXCloseError("Endpoint closed")
        if not isinstance(buffer, Array):
            buffer = Array(buffer)
        if tag is None:
            tag = self._tags["msg_send"]
        elif not force_tag:
            tag = hash64bits(self._tags["msg_send"], hash(tag))
        if not isinstance(tag, Tag):
            tag = Tag(tag)

        # Optimization to eliminate producing logger string overhead
        if logger.isEnabledFor(logging.DEBUG):
            nbytes = buffer.nbytes
            log = "[Send #%03d] ep: 0x%x, tag: 0x%x, nbytes: %d, type: %s" % (
                self._send_count,
                self.uid,
                tag.value,
                nbytes,
                type(buffer.obj),
            )
            logger.debug(log)

        self._send_count += 1

        try:
            request = self._ep.tag_send(buffer, tag)
            return await request.wait()
        except UCXCanceled as e:
            # If self._ep has already been closed and destroyed, we reraise the
            # UCXCanceled exception.
            if self._ep is None:
                raise e

    async def send_multi(self, buffers, tag=None, force_tag=False):
        """Send `buffer` to connected peer.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to send. Raise ValueError if buffer is smaller
            than nbytes.
        tag: hashable, optional
            Set a tag that the receiver must match. Currently the tag
            is hashed together with the internal Endpoint tag that is
            agreed with the remote end at connection time. To enforce
            using the user tag, make sure to specify `force_tag=True`.
        force_tag: bool
            If true, force using `tag` as is, otherwise the value
            specified with `tag` (if any) will be hashed with the
            internal Endpoint tag.
        """
        self._ep.raise_on_error()
        if self.closed:
            raise UCXCloseError("Endpoint closed")
        if not (isinstance(buffers, list) or isinstance(buffers, tuple)):
            raise ValueError("The `buffers` argument must be a `list` or `tuple`")
        buffers = tuple([Array(b) if not isinstance(b, Array) else b for b in buffers])
        if tag is None:
            tag = self._tags["msg_send"]
        elif not force_tag:
            tag = hash64bits(self._tags["msg_send"], hash(tag))
        if not isinstance(tag, Tag):
            tag = Tag(tag)

        # Optimization to eliminate producing logger string overhead
        if logger.isEnabledFor(logging.DEBUG):
            log = "[Send Multi #%03d] ep: 0x%x, tag: 0x%x, nbytes: %s, type: %s" % (
                self._send_count,
                self.uid,
                tag.value,
                tuple([b.nbytes for b in buffers]),  # nbytes,
                tuple([type(b.obj) for b in buffers]),
            )
            logger.debug(log)

        self._send_count += 1

        try:
            buffer_requests = self._ep.tag_send_multi(buffers, tag)
            await buffer_requests.wait()
            buffer_requests.check_error()
        except UCXCanceled as e:
            # If self._ep has already been closed and destroyed, we reraise the
            # UCXCanceled exception.
            if self._ep is None:
                raise e

    async def send_obj(self, obj, tag=None):
        """Send `obj` to connected peer that calls `recv_obj()`.

        The transfer includes an extra message containing the size of `obj`,
        which increases the overhead slightly.

        Parameters
        ----------
        obj: exposing the buffer protocol or array/cuda interface
            The object to send.
        tag: hashable, optional
            Set a tag that the receiver must match.

        Example
        -------
        >>> await ep.send_obj(pickle.dumps([1,2,3]))
        """
        if not isinstance(obj, Array):
            obj = Array(obj)
        nbytes = Array(array.array("Q", [obj.nbytes]))
        await self.send(nbytes, tag=tag)
        await self.send(obj, tag=tag)

    async def am_recv(self):
        """Receive from connected peer via active messages."""
        if not self._ep.am_probe():
            self._ep.raise_on_error()
            if self.closed:
                raise UCXCloseError("Endpoint closed")

        # Optimization to eliminate producing logger string overhead
        if logger.isEnabledFor(logging.DEBUG):
            log = "[AM Recv #%03d] ep: 0x%x" % (
                self._recv_count,
                self.uid,
            )
            logger.debug(log)

        self._recv_count += 1

        req = self._ep.am_recv()
        await req.wait()
        buffer = req.recv_buffer

        if logger.isEnabledFor(logging.DEBUG):
            log = "[AM Recv Completed #%03d] ep: 0x%x, nbytes: %d, type: %s" % (
                self._recv_count,
                self.uid,
                buffer.nbytes,
                type(buffer),
            )
            logger.debug(log)

        self._finished_recv_count += 1
        if (
            self._close_after_n_recv is not None
            and self._finished_recv_count >= self._close_after_n_recv
        ):
            self.abort()
        return buffer

    # @ucx_api.nvtx_annotate("UCXPY_RECV", color="red", domain="ucxpy")
    async def recv(self, buffer, tag=None, force_tag=False):
        """Receive from connected peer into `buffer`.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to receive into. Raise ValueError if buffer
            is smaller than nbytes or read-only.
        tag: hashable, optional
            Set a tag that must match the received message. Currently
            the tag is hashed together with the internal Endpoint tag
            that is agreed with the remote end at connection time.
            To enforce using the user tag, make sure to specify
            `force_tag=True`.
        force_tag: bool
            If true, force using `tag` as is, otherwise the value
            specified with `tag` (if any) will be hashed with the
            internal Endpoint tag.
        """
        if tag is None:
            tag = self._tags["msg_recv"]
        elif not force_tag:
            tag = hash64bits(self._tags["msg_recv"], hash(tag))
        if not isinstance(tag, Tag):
            tag = Tag(tag)

        try:
            self._ep.raise_on_error()
            if self.closed:
                raise UCXCloseError("Endpoint closed")
        except Exception as e:
            # Only probe the worker as last resort. To be reliable, probing for the tag
            # requires progressing the worker, thus prevent that happening too often.
            if not self._ctx.worker.tag_probe(tag):
                raise e

        if not isinstance(buffer, Array):
            buffer = Array(buffer)

        # Optimization to eliminate producing logger string overhead
        if logger.isEnabledFor(logging.DEBUG):
            nbytes = buffer.nbytes
            log = "[Recv #%03d] ep: 0x%x, tag: 0x%x, nbytes: %d, type: %s" % (
                self._recv_count,
                self.uid,
                tag.value,
                nbytes,
                type(buffer.obj),
            )
            logger.debug(log)

        self._recv_count += 1

        req = self._ep.tag_recv(buffer, tag, TagMaskFull)
        ret = await req.wait()

        self._finished_recv_count += 1
        if (
            self._close_after_n_recv is not None
            and self._finished_recv_count >= self._close_after_n_recv
        ):
            self.abort()
        return ret

    async def recv_multi(self, tag=None, force_tag=False):
        """Receive from connected peer into `buffer`.

        Parameters
        ----------
        tag: hashable, optional
            Set a tag that must match the received message. Currently
            the tag is hashed together with the internal Endpoint tag
            that is agreed with the remote end at connection time.
            To enforce using the user tag, make sure to specify
            `force_tag=True`.
        force_tag: bool
            If true, force using `tag` as is, otherwise the value
            specified with `tag` (if any) will be hashed with the
            internal Endpoint tag.
        """
        if tag is None:
            tag = self._tags["msg_recv"]
        elif not force_tag:
            tag = hash64bits(self._tags["msg_recv"], hash(tag))
        if not isinstance(tag, Tag):
            tag = Tag(tag)

        try:
            self._ep.raise_on_error()
            if self.closed:
                raise UCXCloseError("Endpoint closed")
        except Exception as e:
            # Only probe the worker as last resort. To be reliable, probing for the tag
            # requires progressing the worker, thus prevent that happening too often.
            if not self._ctx.worker.tag_probe(tag):
                raise e

        # Optimization to eliminate producing logger string overhead
        if logger.isEnabledFor(logging.DEBUG):
            log = "[Recv Multi #%03d] ep: 0x%x, tag: 0x%x" % (
                self._recv_count,
                self.uid,
                tag.value,
            )
            logger.debug(log)

        self._recv_count += 1

        buffer_requests = self._ep.tag_recv_multi(tag, TagMaskFull)
        await buffer_requests.wait()
        buffer_requests.check_error()
        for r in buffer_requests.requests:
            r.check_error()
        buffers = buffer_requests.py_buffers

        self._finished_recv_count += 1
        if (
            self._close_after_n_recv is not None
            and self._finished_recv_count >= self._close_after_n_recv
        ):
            self.abort()
        return buffers

    async def recv_obj(self, tag=None, allocator=bytearray):
        """Receive from connected peer that calls `send_obj()`.

        As opposed to `recv()`, this function returns the received object.
        Data is received into a buffer allocated by `allocator`.

        The transfer includes an extra message containing the size of `obj`,
        which increases the overhead slightly.

        Parameters
        ----------
        tag: hashable, optional
            Set a tag that must match the received message. Notice, currently
            UCX-Py doesn't support a "any tag" thus `tag=None` only matches a
            send that also sets `tag=None`.
        allocator: callabale, optional
            Function to allocate the received object. The function should
            take the number of bytes to allocate as input and return a new
            buffer of that size as output.

        Example
        -------
        >>> await pickle.loads(ep.recv_obj())
        """
        nbytes = array.array("Q", [0])
        await self.recv(nbytes, tag=tag)
        nbytes = nbytes[0]
        ret = allocator(nbytes)
        await self.recv(ret, tag=tag)
        return ret

    def get_ucp_worker(self):
        """Returns the underlying UCP worker handle (ucp_worker_h)
        as a Python integer.
        """
        warnings.warn(
            "Endpoint.get_ucp_worker() is deprecated and will soon be removed, "
            "use the Endpoint.ucp_worker property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.ucp_worker

    def get_ucxx_worker(self):
        """Returns the underlying UCXX worker pointer (ucxx::Worker*)
        as a Python integer.
        """
        warnings.warn(
            "Endpoint.get_ucxx_worker() is deprecated and will soon be removed, "
            "use the Endpoint.ucxx_worker property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.ucxx_worker

    def get_ucp_endpoint(self):
        """Returns the underlying UCP endpoint handle (ucp_ep_h)
        as a Python integer.
        """
        warnings.warn(
            "Endpoint.get_ucp_endpoint() is deprecated and will soon be removed, "
            "use the Endpoint.ucp_endpoint property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.ucp_endpoint

    def get_ucxx_endpoint(self):
        """Returns the underlying UCXX endpoint pointer (ucxx::Endpoint*)
        as a Python integer.
        """
        warnings.warn(
            "Endpoint.get_ucxx_endpoint() is deprecated and will soon be removed, "
            "use the Endpoint.ucxx_endpoint property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.ucxx_endpoint

    def close_after_n_recv(self, n, count_from_ep_creation=False):
        """Close the endpoint after `n` received messages.

        Parameters
        ----------
        n: int
            Number of messages to received before closing the endpoint.
        count_from_ep_creation: bool, optional
            Whether to count `n` from this function call (default) or
            from the creation of the endpoint.
        """
        if not count_from_ep_creation:
            n += self._finished_recv_count  # Make `n` absolute
        if self._close_after_n_recv is not None:
            raise UCXError(
                "close_after_n_recv has already been set to: %d (abs)"
                % self._close_after_n_recv
            )
        if n == self._finished_recv_count:
            self.abort()
        elif n > self._finished_recv_count:
            self._close_after_n_recv = n
        else:
            raise UCXError(
                "`n` cannot be less than current recv_count: %d (abs) < %d (abs)"
                % (n, self._finished_recv_count)
            )

    def set_close_callback(self, callback_func, cb_args=None, cb_kwargs=None):
        """Register a user callback function to be called on Endpoint's closing.

        Allows the user to register a callback function to be called when the
        Endpoint's error callback is called, or during its finalizer if the error
        callback is never called.

        Once the callback is called, it's not possible to send any more messages.
        However, receiving messages may still be possible, as UCP may still have
        incoming messages in transit.

        Parameters
        ----------
        callback_func: callable
            The callback function to be called when the Endpoint's error callback
            is called, otherwise called on its finalizer.
        cb_args: tuple or None
            The arguments to be passed to the callback function as a `tuple`, or
            `None` (default).
        cb_kwargs: dict or None
            The keyword arguments to be passed to the callback function as a
            `dict`, or `None` (default).

        Example
        >>> ep.set_close_callback(lambda: print("Executing close callback"))
        """
        self._ep.set_close_callback(callback_func, cb_args, cb_kwargs)

    def is_alive(self):
        warnings.warn(
            "Endpoint.is_alive() is deprecated and will soon be removed, "
            "use the Endpoint.alive property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.alive
