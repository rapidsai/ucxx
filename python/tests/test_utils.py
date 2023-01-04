import multiprocessing
import re
from multiprocessing.queues import Empty

import pytest
from utils import terminate_process

# mp = mp.get_context("spawn")
# mp = mp.get_context("forkserver")


def _test_process(queue):
    while True:
        try:
            message = queue.get_nowait()
            assert message == "terminate"
            return
        except Empty:
            pass


@pytest.mark.parametrize("mp_context", ["default", "fork", "forkserver", "spawn"])
def test_terminate_process_clean(mp_context):
    mp = (
        multiprocessing
        if mp_context == "default"
        else multiprocessing.get_context(mp_context)
    )

    queue = mp.Queue()
    proc = mp.Process(
        target=_test_process,
        args=(queue,),
    )
    proc.start()

    queue.put("terminate")

    proc.join()
    terminate_process(proc)


@pytest.mark.parametrize("mp_context", ["default", "fork", "forkserver", "spawn"])
def test_terminate_process_join_timeout(mp_context):
    mp = (
        multiprocessing
        if mp_context == "default"
        else multiprocessing.get_context(mp_context)
    )

    queue = mp.Queue()
    proc = mp.Process(
        target=_test_process,
        args=(queue,),
    )

    proc.start()
    proc.join(timeout=1)

    with pytest.raises(
        RuntimeError, match=re.escape("Process did not exit cleanly (exit code: -9)")
    ):
        terminate_process(proc)


@pytest.mark.parametrize("mp_context", ["default", "fork", "forkserver", "spawn"])
def test_terminate_process_kill_timeout(mp_context):
    mp = (
        multiprocessing
        if mp_context == "default"
        else multiprocessing.get_context(mp_context)
    )

    queue = mp.Queue()
    proc = mp.Process(
        target=_test_process,
        args=(queue,),
    )
    proc.start()
    proc.join(timeout=1)

    with pytest.raises(
        ValueError, match="Cannot close a process while it is still running.*"
    ):
        terminate_process(proc, kill_wait=0.0)
