# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.

import logging
import struct

from ucxx._lib.arr import Array

from .utils import hash64bits

logger = logging.getLogger("ucx")


async def exchange_peer_info(endpoint, msg_tag, ctrl_tag, listener):
    """Help function that exchange endpoint information"""

    # Pack peer information incl. a checksum
    fmt = "QQQ"
    my_info = struct.pack(fmt, msg_tag, ctrl_tag, hash64bits(msg_tag, ctrl_tag))
    peer_info = bytearray(len(my_info))
    my_info_arr = Array(my_info)
    peer_info_arr = Array(peer_info)

    # Send/recv peer information. Notice, we force an `await` between the two
    # streaming calls (see <https://github.com/rapidsai/ucx-py/pull/509>)
    if listener is True:
        req = endpoint.stream_send(my_info_arr)
        await req.wait()
        req = endpoint.stream_recv(peer_info_arr)
        await req.wait()
    else:
        req = endpoint.stream_recv(peer_info_arr)
        await req.wait()
        req = endpoint.stream_send(my_info_arr)
        await req.wait()

    # Unpacking and sanity check of the peer information
    ret = {}
    (ret["msg_tag"], ret["ctrl_tag"], ret["checksum"]) = struct.unpack(fmt, peer_info)

    expected_checksum = hash64bits(ret["msg_tag"], ret["ctrl_tag"])

    if expected_checksum != ret["checksum"]:
        raise RuntimeError(
            f'Checksum invalid! {hex(expected_checksum)} != {hex(ret["checksum"])}'
        )

    return ret
