# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ucxx
from ucxx._lib_async.utils_test import wait_listener_client_handlers


def _primary_context_active(device_id=0):
    from cuda.bindings import driver

    (err,) = driver.cuInit(0)
    assert err == driver.CUresult.CUDA_SUCCESS, err
    err, dev = driver.cuDeviceGet(device_id)
    assert err == driver.CUresult.CUDA_SUCCESS, err
    err, _, active = driver.cuDevicePrimaryCtxGetState(dev)
    assert err == driver.CUresult.CUDA_SUCCESS, err
    return bool(active)


@pytest.mark.asyncio
async def test_cuda_primary_context_retained_across_reset(
    ucxx_retain_cuda_primary_context,
):
    """The session fixture keeps the primary context alive across `ucxx.reset()`.

    In thread progress mode, progress threads must only retain the existing primary
    context, rather than re-creating it for each test after the previous progress
    thread released it, see `ucxx_retain_cuda_primary_context`.
    """
    if not ucxx_retain_cuda_primary_context:
        pytest.skip("CUDA is unavailable")

    assert _primary_context_active()
    for _ in range(2):

        async def server_handler(ep):
            msg = bytearray(1)
            await ep.recv(msg)
            await ep.send(msg)
            await ep.close()

        listener = ucxx.create_listener(server_handler)
        # In thread progress mode, endpoint creation and transfers are serviced by
        # the progress thread, thus it already ran its start callback.
        ep = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
        msg = bytearray(b"x")
        await ep.send(msg)
        await ep.recv(msg)
        await ep.close()
        listener.close()
        await wait_listener_client_handlers(listener)
        del ep, listener

        # Stops (joins) the progress thread
        ucxx.reset()
        assert _primary_context_active()
