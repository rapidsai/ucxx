# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from ucxx._lib_async.utils_test import wait_listener_client_handlers

import ucxx


@pytest.mark.asyncio
async def test_get_ucp_worker():
    worker = ucxx.get_ucp_worker()
    assert isinstance(worker, int)

    async def server(ep):
        assert ep.get_ucp_worker() == worker

    lt = ucxx.create_listener(server)
    ep = await ucxx.create_endpoint(ucxx.get_address(), lt.port)
    assert ep.get_ucp_worker() == worker
    await ep.close()
    await wait_listener_client_handlers(lt)


@pytest.mark.asyncio
async def test_get_endpoint():
    async def server(ep):
        ucp_ep = ep.get_ucp_endpoint()
        assert isinstance(ucp_ep, int)
        assert ucp_ep > 0

    lt = ucxx.create_listener(server)
    ep = await ucxx.create_endpoint(ucxx.get_address(), lt.port)
    ucp_ep = ep.get_ucp_endpoint()
    assert isinstance(ucp_ep, int)
    assert ucp_ep > 0
    await ep.close()
    await wait_listener_client_handlers(lt)
