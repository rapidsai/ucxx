# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


import asyncio
import logging
import struct

from ucxx._lib.arr import Array

from .utils import hash64bits

logger = logging.getLogger("ucx")


async def exchange_peer_info(endpoint, msg_tag, ctrl_tag, listener, stream_timeout=5.0):
    """Help function that exchange endpoint information"""

    # Pack peer information incl. a checksum
    fmt = "QQQ"
    my_info = struct.pack(fmt, msg_tag, ctrl_tag, hash64bits(msg_tag, ctrl_tag))
    my_info_arr = Array(my_info)

    # Send/recv peer information. Notice, we force an `await` between the two
    # streaming calls (see <https://github.com/rapidsai/ucx-py/pull/509>)
    if listener is True:
        req = endpoint.am_send(my_info_arr)
        await asyncio.wait_for(req.wait(), timeout=stream_timeout)
        req = endpoint.am_recv()
        await asyncio.wait_for(req.wait(), timeout=stream_timeout)
        peer_info = req.recv_buffer
    else:
        req = endpoint.am_recv()
        await asyncio.wait_for(req.wait(), timeout=stream_timeout)
        peer_info = req.recv_buffer
        req = endpoint.am_send(my_info_arr)
        await asyncio.wait_for(req.wait(), timeout=stream_timeout)

    # Unpacking and sanity check of the peer information
    ret = {}
    (ret["msg_tag"], ret["ctrl_tag"], ret["checksum"]) = struct.unpack(fmt, peer_info)

    expected_checksum = hash64bits(ret["msg_tag"], ret["ctrl_tag"])

    if expected_checksum != ret["checksum"]:
        raise RuntimeError(
            f'Checksum invalid! {hex(expected_checksum)} != {hex(ret["checksum"])}'
        )

    return ret
