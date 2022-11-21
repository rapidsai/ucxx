import asyncio

import pytest

import ucxx as ucp


@pytest.mark.asyncio
@pytest.mark.parametrize("server_close_callback", [True, False])
async def test_close_callback(server_close_callback):
    closed = [False]

    def _close_callback():
        closed[0] = True

    async def server_node(ep):
        if server_close_callback is True:
            ep.set_close_callback(_close_callback)

    async def client_node(port):
        ep = await ucp.create_endpoint(
            ucp.get_address(),
            port,
        )
        if server_close_callback is False:
            ep.set_close_callback(_close_callback)

    listener = ucp.create_listener(
        server_node,
    )
    await client_node(listener.port)
    while closed[0] is False:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("transfer_api", ["am", "tag"])
async def test_cancel(transfer_api):
    if transfer_api == "am":
        pytest.skip("AM not implemented yet")

    async def server_node(ep):
        await ep.close()

    async def client_node(port):
        ep = await ucp.create_endpoint(ucp.get_address(), port)
        if transfer_api == "am":
            with pytest.raises(
                ucp.exceptions.UCXCanceled,
                # TODO: Add back custom UCXCanceled messages?
            ):
                await ep.am_recv()
        else:
            with pytest.raises(
                ucp.exceptions.UCXCanceled,
                # TODO: Add back custom UCXCanceled messages?
            ):
                msg = bytearray(1)
                await ep.recv(msg)
        await ep.close()

    listener = ucp.create_listener(server_node)
    await client_node(listener.port)
