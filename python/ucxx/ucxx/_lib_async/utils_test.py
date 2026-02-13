# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import io
import logging
import os
from contextlib import contextmanager

import numpy as np
import pytest

import ucxx

normal_env = {
    "UCX_RNDV_SCHEME": "put_zcopy",
    "UCX_MEMTYPE_CACHE": "n",
    "UCX_TLS": "rc,cuda_copy,cuda_ipc",
    "CUDA_VISIBLE_DEVICES": "0",
}


def set_env():
    os.environ.update(normal_env)


def get_num_gpus():
    import pynvml

    pynvml.nvmlInit()
    ngpus = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    return ngpus


def get_cuda_devices():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        ngpus = get_num_gpus()
        return list(range(ngpus))


def compute_timeouts(pytestconfig: pytest.Config) -> tuple[float, float]:
    """
    Calculate low and high timeouts.

    The purpose of those timeouts is ensuring internal tasks can have timeouts
    adjusted based on the total test timeout, for example using low for async
    tasks and high for subprocesses, ensuring timeouts occur in the order: low
    timeout, high timeout and finally test timeout. This is useful to preserve
    information such as async stack and the process that timed out, this can aid
    in resolving issues.

    Parameters
    ----------
    pytestconfig : pytestconfig
        The pytestconfig object retrieved by the object when the fixture with
        same name is added as argument to that function.

    Returns
    -------
    tuple: floats
        Element 0 is the low timeout, and element 1 is the high timeout.
    """
    plugin_timeout = pytestconfig.cache.get("asyncio_timeout", {})["timeout"]
    async_timeout = max(plugin_timeout * 0.8, plugin_timeout - 10)
    join_timeout = max(plugin_timeout * 0.9, plugin_timeout - 5)

    return (async_timeout, join_timeout)


@contextmanager
def captured_logger(logger, level=logging.INFO, propagate=None):
    """Capture output from the given Logger."""
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    orig_level = logger.level
    orig_handlers = logger.handlers[:]
    if propagate is not None:
        orig_propagate = logger.propagate
        logger.propagate = propagate
    sio = io.StringIO()
    logger.handlers[:] = [logging.StreamHandler(sio)]
    logger.setLevel(level)
    try:
        yield sio
    finally:
        logger.handlers[:] = orig_handlers
        logger.setLevel(orig_level)
        if propagate is not None:
            logger.propagate = orig_propagate


def cuda_array(size):
    try:
        import rmm

        return rmm.DeviceBuffer(size=size)
    except ImportError:
        import numba.cuda

        return numba.cuda.device_array((size,), dtype="u1")


async def send(ep, frames):
    pytest.importorskip("distributed")
    from distributed.utils import nbytes

    await ep.send(np.array([len(frames)], dtype=np.uint64))
    await ep.send(
        np.array([hasattr(f, "__cuda_array_interface__") for f in frames], dtype=bool)
    )
    await ep.send(np.array([nbytes(f) for f in frames], dtype=np.uint64))
    # Send frames
    for frame in frames:
        if nbytes(frame) > 0:
            await ep.send(frame)


async def recv(ep):
    pytest.importorskip("distributed")

    from distributed.comm.utils import from_frames

    try:
        # Recv meta data
        nframes = np.empty(1, dtype=np.uint64)
        await ep.recv(nframes)
        is_cudas = np.empty(nframes[0], dtype=bool)
        await ep.recv(is_cudas)
        sizes = np.empty(nframes[0], dtype=np.uint64)
        await ep.recv(sizes)
    except (ucxx.exceptions.UCXCanceledError, ucxx.exceptions.UCXCloseError) as e:
        msg = "SOMETHING TERRIBLE HAS HAPPENED IN THE TEST"
        raise e(msg)

    # Recv frames
    frames = []
    for is_cuda, size in zip(is_cudas.tolist(), sizes.tolist()):
        if size > 0:
            if is_cuda:
                frame = cuda_array(size)
            else:
                frame = np.empty(size, dtype=np.uint8)
            await ep.recv(frame)
            frames.append(frame)
        else:
            if is_cuda:
                frames.append(cuda_array(size))
            else:
                frames.append(b"")

    msg = await from_frames(frames)
    return frames, msg


async def am_send(ep, frames):
    await ep.am_send(np.array([len(frames)], dtype=np.uint64))
    # Send frames
    for frame in frames:
        await ep.am_send(frame)


async def am_recv(ep):
    pytest.importorskip("distributed")

    from distributed.comm.utils import from_frames

    try:
        # Recv meta data
        nframes = (await ep.am_recv()).view(np.uint64)
    except (ucxx.exceptions.UCXCanceledError, ucxx.exceptions.UCXCloseError) as e:
        msg = "SOMETHING TERRIBLE HAS HAPPENED IN THE TEST"
        raise e(msg)

    # Recv frames
    frames = []
    for _ in range(nframes[0]):
        frame = await ep.am_recv()
        frames.append(frame)

    msg = await from_frames(frames)
    return frames, msg


async def wait_listener_client_handlers(listener):
    while listener.active_clients > 0:
        # Minimal delay to yield to the event loop so call_soon_threadsafe callbacks
        # run. Using a very short positive sleep ensures pending callbacks are
        # processed and significantly reduces "coroutine never awaited" warnings.
        await asyncio.sleep(1e-9)
        if not ucxx.core._get_ctx().progress_mode.startswith("thread"):
            ucxx.progress()
