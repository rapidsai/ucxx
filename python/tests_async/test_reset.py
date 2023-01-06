import pytest
from utils import wait_listener_client_handlers

import ucxx as ucp


class ResetAfterN:
    """Calls ucp.reset() after n calls"""

    def __init__(self, n):
        self.n = n
        self.count = 0

    def __call__(self):
        self.count += 1
        if self.count == self.n:
            ucp.reset()


@pytest.mark.asyncio
async def test_reset():
    reset = ResetAfterN(2)

    def server(ep):
        ep.abort()
        reset()

    lt = ucp.create_listener(server)
    ep = await ucp.create_endpoint(ucp.get_address(), lt.port)
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

    lt = ucp.create_listener(server)
    ep = await ucp.create_endpoint(ucp.get_address(), lt.port)
    del ep
    with pytest.raises(
        ucp.exceptions.UCXError,
        match="Trying to reset UCX but not all Endpoints and/or Listeners are closed()",
    ):
        ucp.reset()

    await wait_listener_client_handlers(lt)
    lt.close()
    ucp.reset()


@pytest.mark.asyncio
async def test_ep_still_in_scope_error():
    reset = ResetAfterN(2)

    def server(ep):
        ep.abort()
        reset()

    lt = ucp.create_listener(server)
    ep = await ucp.create_endpoint(ucp.get_address(), lt.port)
    await wait_listener_client_handlers(lt)
    del lt
    with pytest.raises(
        ucp.exceptions.UCXError,
        match="Trying to reset UCX but not all Endpoints and/or Listeners are closed()",
    ):
        ucp.reset()
    ep.abort()
    ucp.reset()
