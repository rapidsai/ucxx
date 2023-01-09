# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

import asyncio
import logging
import os
import threading
import weakref
from queue import Queue

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array

from .continuous_ucx_progress import PollingMode, ThreadMode
from .endpoint import Endpoint
from .exchange_peer_info import exchange_peer_info
from .listener import ActiveClients, Listener, _listener_handler
from .notifier_thread import _notifierThread
from .utils import get_event_loop, hash64bits

logger = logging.getLogger("ucx")


class ApplicationContext:
    """
    The context of the Asyncio interface of UCX.
    """

    def __init__(
        self,
        config_dict={},
        progress_mode=None,
        enable_delayed_submission=None,
        enable_python_future=None,
    ):
        self.progress_tasks = []
        self.notifier_thread_q = None
        self.notifier_thread = None
        self._listener_active_clients = ActiveClients()
        self._next_listener_id = 0

        self.progress_mode = ApplicationContext._check_progress_mode(progress_mode)

        enable_delayed_submission = ApplicationContext._check_enable_delayed_submission(
            enable_delayed_submission
        )
        enable_python_future = ApplicationContext._check_enable_python_future(
            enable_python_future, self.progress_mode
        )

        # For now, a application context only has one worker
        self.context = ucx_api.UCXContext(config_dict)
        self.worker = ucx_api.UCXWorker(
            self.context,
            enable_delayed_submission=enable_delayed_submission,
            enable_python_future=enable_python_future,
        )

        self.start_notifier_thread()

        weakref.finalize(self, self.progress_tasks.clear)

        # Ensure progress even before Endpoints get created, for example to
        # receive messages directly on a worker after a remote endpoint
        # connected with `create_endpoint_from_worker_address`.
        self.continuous_ucx_progress()

    @staticmethod
    def _check_progress_mode(progress_mode):
        if progress_mode is None:
            if "UCXPY_PROGRESS_MODE" in os.environ:
                progress_mode = os.environ["UCXPY_PROGRESS_MODE"]
            else:
                progress_mode = "thread"

        valid_progress_modes = ["polling", "thread", "thread-polling"]
        if not isinstance(progress_mode, str) or not any(
            progress_mode == m for m in valid_progress_modes
        ):
            raise ValueError(
                f"Unknown progress mode {progress_mode}, "
                "valid modes are: 'blocking', 'polling', 'thread' or 'thread-polling'"
            )

        return progress_mode

    @staticmethod
    def _check_enable_delayed_submission(enable_delayed_submission):
        if enable_delayed_submission is None:
            if "UCXPY_ENABLE_DELAYED_SUBMISSION" in os.environ:
                enable_delayed_submission = (
                    False
                    if os.environ["UCXPY_ENABLE_DELAYED_SUBMISSION"] == "0"
                    else True
                )
            else:
                enable_delayed_submission = True

    @staticmethod
    def _check_enable_python_future(enable_python_future, progress_mode):
        if enable_python_future is None:
            if "UCXPY_ENABLE_PYTHON_FUTURE" in os.environ:
                explicit_enable_python_future = (
                    os.environ["UCXPY_ENABLE_PYTHON_FUTURE"] != "0"
                )
            else:
                explicit_enable_python_future = False
        else:
            explicit_enable_python_future = enable_python_future

        if not progress_mode.startswith("thread") and explicit_enable_python_future:
            logger.warning(
                f"Notifier thread requested, but {progress_mode} does not "
                "support it, using Python wait_yield()."
            )
            enable_python_future = False
        return explicit_enable_python_future

    def start_notifier_thread(self):
        if self.worker.is_python_future_enabled():
            logger.debug("UCXX_ENABLE_PYTHON available, enabling notifier thread")
            loop = get_event_loop()
            self.notifier_thread_q = Queue()
            self.notifier_thread = threading.Thread(
                target=_notifierThread,
                args=(loop, self.worker, self.notifier_thread_q),
                name="UCX-Py Async Notifier Thread",
            )
            self.notifier_thread.start()
        else:
            logger.debug(
                "UCXX not compiled with UCXX_ENABLE_PYTHON, disabling notifier thread"
            )

    def stop_notifier_thread(self):
        """
        Stop Python future notifier thread

        Stop the notifier thread if context is running with Python future
        notification enabled via `UCXPY_ENABLE_PYTHON_FUTURE=1` or
        `ucp.init(..., enable_python_future=True)`.

        .. warning:: When the notifier thread is enabled it may be necessary to
                     explicitly call this method before shutting down the process or
                     or application, otherwise it may block indefinitely waiting for
                     the thread to terminate. Executing `ucp.reset()` will also run
                     this method, so it's not necessary to have both.
        """
        if self.notifier_thread_q and self.notifier_thread:
            self.notifier_thread_q.put("shutdown")
            while True:
                # Having a timeout is required. During the notifier thread shutdown
                # it may require the GIL, which will cause a deadlock with the `join()`
                # call otherwise.
                self.notifier_thread.join(timeout=0.01)
                if not self.notifier_thread.is_alive():
                    break
            logger.debug("Notifier thread stopped")
        else:
            logger.debug("Notifier thread not running")

    def create_listener(
        self,
        callback_func,
        port=0,
        endpoint_error_handling=True,
    ):
        """Create and start a listener to accept incoming connections

        callback_func is the function or coroutine that takes one
        argument -- the Endpoint connected to the client.

        Notice, the listening is closed when the returned Listener
        goes out of scope thus remember to keep a reference to the object.

        Parameters
        ----------
        callback_func: function or coroutine
            A callback function that gets invoked when an incoming
            connection is accepted
        port: int, optional
            An unused port number for listening, or `0` to let UCX assign
            an unused port.
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling raising
            exceptions when an error occurs, may incur in performance penalties
            but prevents a process from terminating unexpectedly that may
            happen when disabled. If `False` endpoint endpoint error handling
            is disabled.

        Returns
        -------
        Listener
            The new listener. When this object is deleted, the listening stops
        """
        self.continuous_ucx_progress()
        if port is None:
            port = 0

        loop = get_event_loop()

        logger.info("create_listener() - Start listening on port %d" % port)
        listener_id = self._next_listener_id
        self._next_listener_id += 1
        ret = Listener(
            ucx_api.UCXListener.create(
                worker=self.worker,
                port=port,
                cb_func=_listener_handler,
                cb_args=(
                    loop,
                    callback_func,
                    self,
                    endpoint_error_handling,
                    listener_id,
                    self._listener_active_clients,
                ),
                deliver_endpoint=True,
            ),
            listener_id,
            self._listener_active_clients,
        )
        return ret

    async def create_endpoint(self, ip_address, port, endpoint_error_handling=True):
        """Create a new endpoint to a server

        Parameters
        ----------
        ip_address: str
            IP address of the server the endpoint should connect to
        port: int
            IP address of the server the endpoint should connect to
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling raising
            exceptions when an error occurs, may incur in performance penalties
            but prevents a process from terminating unexpectedly that may
            happen when disabled. If `False` endpoint endpoint error handling
            is disabled.

        Returns
        -------
        Endpoint
            The new endpoint
        """
        self.continuous_ucx_progress()

        ucx_ep = ucx_api.UCXEndpoint.create(
            self.worker, ip_address, port, endpoint_error_handling
        )
        self.worker.progress()

        # We create the Endpoint in three steps:
        #  1) Generate unique IDs to use as tags
        #  2) Exchange endpoint info such as tags
        #  3) Use the info to create an endpoint
        seed = os.urandom(16)
        msg_tag = hash64bits("msg_tag", seed, ucx_ep.handle)
        ctrl_tag = hash64bits("ctrl_tag", seed, ucx_ep.handle)
        peer_info = await exchange_peer_info(
            endpoint=ucx_ep,
            msg_tag=msg_tag,
            ctrl_tag=ctrl_tag,
            listener=False,
        )
        tags = {
            "msg_send": peer_info["msg_tag"],
            "msg_recv": msg_tag,
            "ctrl_send": peer_info["ctrl_tag"],
            "ctrl_recv": ctrl_tag,
        }
        ep = Endpoint(endpoint=ucx_ep, ctx=self, tags=tags)

        logger.debug(
            "create_endpoint() client: %s, error handling: %s, msg-tag-send: %s, "
            "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s"
            % (
                hex(ep._ep.handle),
                endpoint_error_handling,
                hex(ep._tags["msg_send"]),
                hex(ep._tags["msg_recv"]),
                hex(ep._tags["ctrl_send"]),
                hex(ep._tags["ctrl_recv"]),
            )
        )

        return ep

    async def create_endpoint_from_worker_address(
        self,
        address,
        endpoint_error_handling=True,
    ):
        """Create a new endpoint to a server

        Parameters
        ----------
        address: UCXAddress
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling raising
            exceptions when an error occurs, may incur in performance penalties
            but prevents a process from terminating unexpectedly that may
            happen when disabled. If `False` endpoint endpoint error handling
            is disabled.

        Returns
        -------
        Endpoint
            The new endpoint
        """
        self.continuous_ucx_progress()

        ucx_ep = ucx_api.UCXEndpoint.create_from_worker_address(
            self.worker,
            address,
            endpoint_error_handling,
        )
        self.worker.progress()

        ep = Endpoint(endpoint=ucx_ep, ctx=self, tags=None)

        logger.debug(
            "create_endpoint() client: %s, error handling: %s"
            % (hex(ep._ep.handle), endpoint_error_handling)
        )

        return ep

    def continuous_ucx_progress(self, event_loop=None):
        """Guarantees continuous UCX progress

        Use this function to associate UCX progress with an event loop.
        Notice, multiple event loops can be associate with UCX progress.

        This function is automatically called when calling
        `create_listener()` or `create_endpoint()`.

        Parameters
        ----------
        event_loop: asyncio.event_loop, optional
            The event loop to evoke UCX progress. If None,
            `asyncio.get_event_loop()` (`asyncio.new_event_loop()` in
            Python 3.10+) is used.
        """
        loop = event_loop if event_loop is not None else get_event_loop()
        if loop in self.progress_tasks:
            return  # Progress has already been guaranteed for the current event loop

        if self.progress_mode == "thread":
            task = ThreadMode(self.worker, loop, polling_mode=False)
        elif self.progress_mode == "thread-polling":
            task = ThreadMode(self.worker, loop, polling_mode=True)
        elif self.progress_mode == "polling":
            task = PollingMode(self.worker, loop)

        self.progress_tasks.append(task)

    def get_ucp_worker(self):
        """Returns the underlying UCP worker handle (ucp_worker_h)
        as a Python integer.
        """
        return self.worker.handle

    def get_config(self):
        """Returns all UCX configuration options as a dict.

        Returns
        -------
        dict
            The current UCX configuration options
        """
        return self.context.get_config()

    def ucp_context_info(self):
        """Return low-level UCX info about this endpoint as a string"""
        return self.context.info

    def ucp_worker_info(self):
        """Return low-level UCX info about this endpoint as a string"""
        return self.worker.info

    def get_worker_address(self):
        return self.worker.get_address()

    # @ucx_api.nvtx_annotate("UCXPY_WORKER_RECV", color="red", domain="ucxpy")
    async def recv(self, buffer, tag):
        """Receive directly on worker without a local Endpoint into `buffer`.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to receive into. Raise ValueError if buffer
            is smaller than nbytes or read-only.
        tag: hashable, optional
            Set a tag that must match the received message.
        """
        if not isinstance(buffer, Array):
            buffer = Array(buffer)
        nbytes = buffer.nbytes
        log = "[Worker Recv] worker: %s, tag: %s, nbytes: %d, type: %s" % (
            hex(self.worker.handle),
            hex(tag),
            nbytes,
            type(buffer.obj),
        )
        logger.debug(log)

        req = self.worker.tag_recv(buffer, tag)
        return await req.wait()
