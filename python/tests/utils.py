import time
from multiprocessing.process import BaseProcess
from typing import Type, Union


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
