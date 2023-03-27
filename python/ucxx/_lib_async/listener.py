# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import logging
import os
import threading

import ucxx._lib.libucxx as ucx_api

from .endpoint import Endpoint
from .exchange_peer_info import exchange_peer_info
from .utils import hash64bits

logger = logging.getLogger("ucx")


class ActiveClients:
    """
    Handle number of active clients on `Listener`.

    Each `Listener` contains a unique ID that can be used to increment/decrement the
    number of currently active client handlers. Useful to warn when the `Listener` is
    being destroyed but callbacks handling clients have not yet completed, which may
    lead to errors as the `Listener` most likely ended prematurely.
    """

    def __init__(self):
        self._locks = dict()
        self._active_clients = dict()

    def add_listener(self, id: int) -> None:
        if id in self._active_clients:
            raise ValueError("Listener {id} is already registered in ActiveClients.")

        self._locks[id] = threading.Lock()
        self._active_clients[id] = 0

    def remove_listener(self, id: int) -> None:
        with self._locks[id]:
            active_clients = self.get_active(id)
            if active_clients > 0:
                raise RuntimeError(
                    "Listener {id} is being removed from ActiveClients, but "
                    f"{active_clients} active client(s) is(are) still accounted for."
                )

        del self._locks[id]
        del self._active_clients[id]

    def inc(self, id: int) -> None:
        with self._locks[id]:
            self._active_clients[id] += 1

    def dec(self, id: int) -> None:
        with self._locks[id]:
            if self._active_clients[id] == 0:
                raise ValueError(f"There are no active clients for listener {id}")
            self._active_clients[id] -= 1

    def get_active(self, id: int) -> int:
        return self._active_clients[id]


class Listener:
    """A handle to the listening service started by `create_listener()`

    The listening continues as long as this object exist or `.close()` is called.
    Please use `create_listener()` to create an Listener.
    """

    def __init__(self, listener, id, active_clients):
        if not isinstance(listener, ucx_api.UCXListener):
            raise ValueError("listener must be an instance of UCXListener")

        self._listener = listener

        active_clients.add_listener(id)
        self._id = id
        self._active_clients = active_clients

    def __del__(self):
        try:
            self._active_clients.remove_listener(self._id)
        except RuntimeError:
            active_clients = self._active_clients.get_active(self._id)
            logger.warning(
                f"Listener object is being destroyed, but {active_clients} client "
                "handler(s) is(are) still alive. This usually indicates the Listener "
                "was prematurely destroyed."
            )

    def closed(self):
        """Is the listener closed?"""
        return self._listener is None

    @property
    def ip(self):
        """The listening network IP address"""
        return self._listener.ip

    @property
    def port(self):
        """The listening network port"""
        return self._listener.port

    @property
    def active_clients(self):
        return self._active_clients.get_active(self._id)

    def close(self):
        """Closing the listener"""
        self._listener = None


async def _listener_handler_coroutine(
    conn_request, ctx, func, endpoint_error_handling, id, active_clients
):
    # def _listener_handler_coroutine(
    #     conn_request, ctx, func, endpoint_error_handling, id, active_clients
    # ):
    # We create the Endpoint in five steps:
    #  1) Create endpoint from conn_request
    #  2) Generate unique IDs to use as tags
    #  3) Exchange endpoint info such as tags
    #  4) Setup control receive callback
    #  5) Execute the listener's callback function
    active_clients.inc(id)
    endpoint = conn_request

    seed = os.urandom(16)
    msg_tag = hash64bits("msg_tag", seed, endpoint.handle)
    ctrl_tag = hash64bits("ctrl_tag", seed, endpoint.handle)

    peer_info = await exchange_peer_info(
        endpoint=endpoint,
        msg_tag=msg_tag,
        ctrl_tag=ctrl_tag,
        listener=True,
    )
    tags = {
        "msg_send": peer_info["msg_tag"],
        "msg_recv": msg_tag,
        "ctrl_send": peer_info["ctrl_tag"],
        "ctrl_recv": ctrl_tag,
    }
    ep = Endpoint(endpoint=endpoint, ctx=ctx, tags=tags)

    logger.debug(
        "_listener_handler() server: %s, error handling: %s, msg-tag-send: %s, "
        "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s"
        % (
            hex(endpoint.handle),
            endpoint_error_handling,
            hex(ep._tags["msg_send"]),
            hex(ep._tags["msg_recv"]),
            hex(ep._tags["ctrl_send"]),
            hex(ep._tags["ctrl_recv"]),
        )
    )

    # Removing references here to avoid delayed clean up
    del ctx

    # Finally, we call `func`
    if asyncio.iscoroutinefunction(func):
        try:
            await func(ep)
        except Exception as e:
            logger.error(f"Uncatched listener callback error {type(e)}: {e}")
    else:
        func(ep)

    active_clients.dec(id)

    # Ensure `ep` is destroyed and `__del__` is called
    del ep


def _listener_handler(
    conn_request,
    event_loop,
    callback_func,
    ctx,
    endpoint_error_handling,
    id,
    active_clients,
):
    asyncio.run_coroutine_threadsafe(
        _listener_handler_coroutine(
            conn_request,
            ctx,
            callback_func,
            endpoint_error_handling,
            id,
            active_clients,
        ),
        event_loop,
    )
