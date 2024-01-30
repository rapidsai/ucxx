# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio

import pytest

from distributed.comm import connect, listen
from distributed.protocol import to_serialize

import ucxx

from distributed_ucxx.utils_test import gen_test, ucxx_loop


try:
    HOST = ucxx.get_address()
except Exception:
    HOST = "127.0.0.1"


async def get_comm_pair(
    listen_addr=f"ucxx://{HOST}", listen_args=None, connect_args=None, **kwargs
):
    listen_args = listen_args or {}
    connect_args = connect_args or {}
    q = asyncio.queues.Queue()

    async def handle_comm(comm):
        await q.put(comm)

    listener = listen(listen_addr, handle_comm, **listen_args, **kwargs)
    async with listener:
        comm = await connect(listener.contact_address, **connect_args, **kwargs)
        serv_comm = await q.get()
        return (comm, serv_comm)


@pytest.mark.parametrize(
    "g",
    [
        lambda cudf: cudf.Series([1, 2, 3]),
        lambda cudf: cudf.DataFrame({"a": [1, 2, None], "b": [1.0, 2.0, None]}),
    ],
)
@gen_test()
async def test_ping_pong_cudf(ucxx_loop, g):
    cudf = pytest.importorskip("cudf")
    from cudf.testing._utils import assert_eq

    cudf_obj = g(cudf)

    com, serv_com = await get_comm_pair()
    msg = {"op": "ping", "data": to_serialize(cudf_obj)}

    await com.write(msg)
    result = await serv_com.read()

    cudf_obj_2 = result.pop("data")
    assert result["op"] == "ping"
    assert_eq(cudf_obj, cudf_obj_2)

    await com.close()
    await serv_com.close()
