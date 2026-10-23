# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import asyncio
import logging
import os
import struct
import time

from ucxx._lib.arr import Array

from .utils import hash64bits

logger = logging.getLogger("ucx")


async def exchange_peer_info(endpoint, msg_tag, listener, connect_timeout=5.0):
    """Help function that exchange endpoint information"""

    # Pack peer information incl. a checksum
    fmt = "QQ"
    my_info = struct.pack(fmt, msg_tag, hash64bits(msg_tag))
    peer_info = bytearray(len(my_info))
    my_info_arr = Array(my_info)
    peer_info_arr = Array(peer_info)
    trace_enabled = os.environ.get("UCXX_BOOTSTRAP_TRACE") == "1"
    stage = "start"
    endpoint_id = hex(endpoint.handle) if trace_enabled else None

    def trace(event, **details):
        if trace_enabled:
            print(
                f"UCXX_BOOTSTRAP_TRACE time_ns={time.monotonic_ns()} "
                f"pid={os.getpid()} role={'listener' if listener else 'client'} "
                f"endpoint={endpoint_id} msg_tag={hex(msg_tag)} "
                f"stage={event} details={details}",
                flush=True,
            )

    # Send/recv peer information. Notice, we force an `await` between the two
    # streaming calls (see <https://github.com/rapidsai/ucx-py/pull/509>)
    try:
        if listener is True:
            stage = "send"
            trace("send-start", length=len(my_info), timeout=connect_timeout)
            req = endpoint.stream_send(my_info_arr)
            await asyncio.wait_for(req.wait(), timeout=connect_timeout)
            trace("send-done", length=len(my_info))
            stage = "recv"
            trace("recv-start", length=len(peer_info), timeout=connect_timeout)
            req = endpoint.stream_recv(peer_info_arr)
            await asyncio.wait_for(req.wait(), timeout=connect_timeout)
            trace("recv-done", length=len(peer_info))
        else:
            stage = "recv"
            trace("recv-start", length=len(peer_info), timeout=connect_timeout)
            req = endpoint.stream_recv(peer_info_arr)
            await asyncio.wait_for(req.wait(), timeout=connect_timeout)
            trace("recv-done", length=len(peer_info))
            stage = "send"
            trace("send-start", length=len(my_info), timeout=connect_timeout)
            req = endpoint.stream_send(my_info_arr)
            await asyncio.wait_for(req.wait(), timeout=connect_timeout)
            trace("send-done", length=len(my_info))

        # Unpacking and sanity check of the peer information
        ret = {}
        (ret["msg_tag"], ret["checksum"]) = struct.unpack(fmt, peer_info)

        expected_checksum = hash64bits(ret["msg_tag"])

        if expected_checksum != ret["checksum"]:
            raise RuntimeError(
                f"Checksum invalid! {hex(expected_checksum)} != {hex(ret['checksum'])}"
            )

        trace("complete", peer_msg_tag=hex(ret["msg_tag"]))
        return ret
    except BaseException as e:
        trace("error", failed_stage=stage, error=repr(e))
        raise
