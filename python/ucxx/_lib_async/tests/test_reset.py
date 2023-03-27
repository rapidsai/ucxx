# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from utils import wait_listener_client_handlers

import ucxx


class ResetAfterN:
    """Calls ucxx.reset() after n calls"""

    def __init__(self, n):
        self.n = n
        self.count = 0

    def __call__(self):
        self.count += 1
        if self.count == self.n:
            ucxx.reset()


@pytest.mark.asyncio
async def test_reset():
    reset = ResetAfterN(2)

    def server(ep):
        ep.abort()
        reset()

    lt = ucxx.create_listener(server)
    ep = await ucxx.create_endpoint(ucxx.get_address(), lt.port)
    await wait_listener_client_handlers(lt)
    del lt
    del ep
    reset()


@pytest.mark.asyncio
async def test_lt_still_in_scope_error():
    reset = ResetAfterN(2)

    def server(ep):
        ep.abort()
        reset()

    lt = ucxx.create_listener(server)
    ep = await ucxx.create_endpoint(ucxx.get_address(), lt.port)
    del ep
    with pytest.raises(
        ucxx.exceptions.UCXError,
        match="Trying to reset UCX but not all Endpoints and/or Listeners are closed()",
    ):
        ucxx.reset()

    await wait_listener_client_handlers(lt)
    lt.close()
    ucxx.reset()


@pytest.mark.asyncio
async def test_ep_still_in_scope_error():
    reset = ResetAfterN(2)

    def server(ep):
        ep.abort()
        reset()

    lt = ucxx.create_listener(server)
    ep = await ucxx.create_endpoint(ucxx.get_address(), lt.port)
    await wait_listener_client_handlers(lt)
    del lt
    with pytest.raises(
        ucxx.exceptions.UCXError,
        match="Trying to reset UCX but not all Endpoints and/or Listeners are closed()",
    ):
        ucxx.reset()
    ep.abort()
    ucxx.reset()
