# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

import asyncio
import logging
import os

import ucxx._lib.libucxx as ucx_api

from .endpoint import Endpoint
from .exchange_peer_info import exchange_peer_info
from .utils import hash64bits

logger = logging.getLogger("ucx")


class Listener:
    """A handle to the listening service started by `create_listener()`

    The listening continues as long as this object exist or `.close()` is called.
    Please use `create_listener()` to create an Listener.
    """

    def __init__(self, listener):
        if not isinstance(listener, ucx_api.UCXListener):
            raise ValueError("listener must be an instance of UCXListener")

        self._listener = listener

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

    def close(self):
        """Closing the listener"""
        self._listener = None


async def _listener_handler_coroutine(conn_request, ctx, func, endpoint_error_handling):
    # def _listener_handler_coroutine(conn_request, ctx, func, endpoint_error_handling):
    # We create the Endpoint in five steps:
    #  1) Create endpoint from conn_request
    #  2) Generate unique IDs to use as tags
    #  3) Exchange endpoint info such as tags
    #  4) Setup control receive callback
    #  5) Execute the listener's callback function
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

    # Ensure `ep` is destroyed and `__del__` is called
    del ep


def _listener_handler(
    conn_request, event_loop, callback_func, ctx, endpoint_error_handling
):
    asyncio.run_coroutine_threadsafe(
        _listener_handler_coroutine(
            conn_request,
            ctx,
            callback_func,
            endpoint_error_handling,
        ),
        event_loop,
    )
