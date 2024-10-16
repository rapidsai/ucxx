# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import time
from multiprocessing.process import BaseProcess
from typing import Type, Union


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
    process: Type[BaseProcess], kill_wait: Union[float, int] = 3.0
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
        raise RuntimeError(
            f"Process did not exit cleanly (exit code: {process.exitcode})"
        )


def wait_requests(worker, progress_mode, requests):
    if not isinstance(requests, list):
        requests = [requests]

    while not all([r.completed for r in requests]):
        if progress_mode == "blocking":
            worker.progress_worker_event()

    for r in requests:
        r.check_error()
