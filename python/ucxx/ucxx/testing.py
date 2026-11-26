# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import queue as _queue_module
import traceback as _traceback_module
import time
import warnings
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as _MPQueue
from typing import Any, Callable, Iterator, Optional, Type, Union


def run_in_subprocess(
    target: Callable[..., Any],
    error_queue: _MPQueue[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Run function in a subprocess, forwarding exceptions to an error queue.

    Use this as the ``target`` for ``multiprocessing.Process``, passing the
    same ``error_queue`` to :func:`terminate_process`.  Any unhandled exception
    raised by ``target`` will be serialized as a formatted traceback string and
    placed in ``error_queue`` before being re-raised, so the process still
    exits with a non-zero code.  :func:`terminate_process` then reads from
    that queue and raises a :class:`RuntimeError` that includes the full
    subprocess traceback rather than the generic "Process did not exit cleanly"
    message.

    Parameters
    ----------
    target : callable
        The function to run in the subprocess.
    error_queue : multiprocessing.Queue
        Queue that receives the formatted traceback string on failure.
    *args
        Positional arguments forwarded to ``target``.
    **kwargs
        Keyword arguments forwarded to ``target``.
    """
    try:
        target(*args, **kwargs)
    except Exception:
        error_queue.put(_traceback_module.format_exc())
        raise


def run_in_subprocess_with_cuda_context(
    target: Callable[..., Any],
    error_queue: _MPQueue[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Like :func:`run_in_subprocess`, retaining the CUDA primary context.

    Retains the primary context of device ``0`` (relative to the subprocess'
    ``CUDA_VISIBLE_DEVICES``) from the subprocess' main thread before running
    ``target``, see :func:`retain_cuda_primary_context`. Use it for subprocesses
    using UCXX in thread progress mode (i.e., ``ucxx`` asyncio API), whose progress
    thread would otherwise create the context while the first endpoint creation
    waits for it, which may take long enough on a loaded GPU for endpoint
    creation to time out. The session fixtures do not apply to subprocesses.

    Parameters
    ----------
    target : callable
        The function to run in the subprocess.
    error_queue : multiprocessing.Queue
        Queue that receives the formatted traceback string on failure.
    *args
        Positional arguments forwarded to ``target``.
    **kwargs
        Keyword arguments forwarded to ``target``.
    """
    with retain_cuda_primary_context():
        run_in_subprocess(target, error_queue, *args, **kwargs)


def join_processes(
    processes: list[Type[BaseProcess]],
    timeout: Union[float, int],
) -> None:
    """
    Join a list of processes with a combined timeout.

    Join a list of processes with a combined timeout, for each process `join()`
    is called with a timeout equal to the difference of `timeout` and the time
    elapsed since this function was called.

    Parameters
    ----------
    processes:
        The list of processes to be joined.
    timeout: float or integer
        Maximum time to wait for all the processes to be joined.
    """
    start = time.monotonic()
    for p in processes:
        t = timeout - (time.monotonic() - start)
        p.join(timeout=t)


def terminate_process(
    process: Type[BaseProcess],
    kill_wait: Union[float, int] = 3.0,
    error_queue: Optional[_MPQueue[str]] = None,
) -> None:
    """
    Ensure a spawned process is terminated.

    Ensure a spawned process is really terminated to prevent the parent process
    (such as pytest) from freezing upon exit.

    Parameters
    ----------
    process:
        The process to be terminated.
    kill_wait: float or integer
        Maximum time to wait for the kill signal to terminate the process.
    error_queue: multiprocessing.Queue, optional
        When the subprocess was started via :func:`run_in_subprocess`, pass
        the same queue here.  If the process exited with a non-zero code,
        ``terminate_process`` will pull the formatted traceback from the queue
        and include it in the raised :class:`RuntimeError`, making the root
        cause visible without digging into subprocess logs.

    Raises
    ------
    RuntimeError
        If the process terminated with a non-zero exit code.
    ValueError
        If the process was still alive after ``kill_wait`` seconds.
    """
    # Ensure process doesn't remain alive and hangs pytest
    if process.is_alive():
        process.kill()

    start_time = time.monotonic()
    while time.monotonic() - start_time < kill_wait:
        if not process.is_alive():
            break

    if process.is_alive():
        process.close()
    elif process.exitcode != 0:
        msg = f"Process did not exit cleanly (exit code: {process.exitcode})"
        if error_queue is not None:
            try:
                tb = error_queue.get_nowait()
                raise RuntimeError(f"{msg}\n\nSubprocess traceback:\n{tb}")
            except _queue_module.Empty:
                pass
        raise RuntimeError(msg)


def wait_requests(worker, progress_mode, requests):
    if not isinstance(requests, list):
        requests = [requests]

    while not all([r.completed for r in requests]):
        if progress_mode == "blocking":
            worker.progress_worker_event()

    for r in requests:
        r.check_error()


@contextlib.contextmanager
def retain_cuda_primary_context(device_id: int = 0) -> Iterator[bool]:
    """Retain the primary CUDA context of a device while the context is active.

    The context is retained with ``cuDevicePrimaryCtxRetain`` without making it
    current on the calling thread, and released with ``cuDevicePrimaryCtxRelease``
    on exit. If CUDA is unavailable (no ``cuda.bindings``, no driver or no visible
    devices), nothing is retained. Any other failure (e.g., the driver failing to
    retain the context) is reported with a ``RuntimeWarning`` and nothing is
    retained.

    This is meant for test suites, see the session fixtures in the ``conftest.py``
    files that use it. In thread progress mode, every progress thread first runs
    ``ucxx._lib_async.continuous_ucx_progress._create_context()``, which makes the
    primary context of device ``0`` current, creating it if no other thread
    retains it, and the last thread retaining it releases (destroys) it when it
    exits. Tests call ``ucxx.reset()`` for each test, thus without another holder
    each test re-creates the context in its new progress thread. Under load from
    other processes on the GPU that can take several seconds, while the main
    thread waits for the progress thread to create endpoints with a limited
    budget, causing spurious ``UCXXTimedOutError``. Retaining the context for the
    lifetime of the test process makes subsequent retains by progress threads
    cheap, without changing the context lifetime in the library itself, which
    applications may rely on.

    Parameters
    ----------
    device_id : int, optional
        The CUDA device ordinal, relative to ``CUDA_VISIBLE_DEVICES`` (default: 0,
        matching the device used by the progress thread start callback).

    Yields
    ------
    bool
        Whether the primary context was retained.
    """
    driver = None
    device = None
    try:
        from cuda.bindings import driver
    except ImportError:
        # No CUDA Python bindings (e.g., a CPU-only environment), nothing to retain.
        pass
    else:
        try:
            device = _retain_cuda_primary_context(driver, device_id)
        except Exception as e:
            warnings.warn(
                f"Failed to retain the CUDA primary context of device {device_id}: "
                f"{e!r}. Endpoint creation in thread progress mode may time out "
                "while the progress thread creates the context.",
                RuntimeWarning,
                stacklevel=3,
            )

    try:
        yield device is not None
    finally:
        if device is not None:
            (err,) = driver.cuDevicePrimaryCtxRelease(device)
            if err != driver.CUresult.CUDA_SUCCESS:
                warnings.warn(
                    f"Failed to release the CUDA primary context of device "
                    f"{device_id}: {err!r}",
                    RuntimeWarning,
                    stacklevel=3,
                )


def _retain_cuda_primary_context(driver: Any, device_id: int) -> Optional[Any]:
    """Retain the primary CUDA context of ``device_id``.

    Returns the ``CUdevice`` whose primary context was retained, or ``None`` if
    CUDA is unavailable (no driver or no visible device). Raises ``RuntimeError``
    on any other CUDA driver error, e.g., if retaining the primary context fails.
    """
    try:
        from cuda.pathfinder import DynamicLibNotFoundError
    except ImportError:
        DynamicLibNotFoundError = ()  # Older `cuda.bindings`, catch nothing.

    try:
        (err,) = driver.cuInit(0)
    except DynamicLibNotFoundError:
        # No NVIDIA driver installed.
        return None
    if err in (
        driver.CUresult.CUDA_ERROR_NO_DEVICE,
        driver.CUresult.CUDA_ERROR_STUB_LIBRARY,
    ):
        # No (visible) CUDA device, e.g., `CUDA_VISIBLE_DEVICES=""`, or only the
        # driver stub library is available.
        return None
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {err!r}")
    err, device = driver.cuDeviceGet(device_id)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet({device_id}) failed: {err!r}")
    err, _ = driver.cuDevicePrimaryCtxRetain(device)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {err!r}")
    return device
