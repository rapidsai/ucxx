# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import asyncio


class ProgressTask(object):
    def __init__(self, worker, event_loop):
        """Creates a task that keeps calling worker.progress()

        Notice, class and created task is carefull not to hold a
        reference to `worker` so that a danling progress task will
        not prevent `worker` to be garbage collected.

        Parameters
        ----------
        worker: UCXWorker
            The UCX worker context to progress
        event_loop: asyncio.EventLoop
            The event loop to do progress in.
        """
        self.worker = worker
        self.event_loop = event_loop
        self.asyncio_task = None

    def __del__(self):
        if self.asyncio_task is not None:
            # FIXME: This does not work, the cancellation must be awaited.
            # Running with polling mode will always cause
            # `Task was destroyed but it is pending!` errors at ucp.reset().
            self.asyncio_task.cancel()

    # Hash and equality is based on the event loop
    def __hash__(self):
        return hash(self.event_loop)

    def __eq__(self, other):
        return hash(self) == hash(other)


def _create_context():
    import numba.cuda

    numba.cuda.current_context()


class ThreadMode(ProgressTask):
    def __init__(self, worker, event_loop, polling_mode=False):
        super().__init__(worker, event_loop)
        worker.set_progress_thread_start_callback(_create_context)
        worker.start_progress_thread(polling_mode=polling_mode)

    def __del__(self):
        self.worker.stop_progress_thread()


class PollingMode(ProgressTask):
    def __init__(self, worker, event_loop):
        super().__init__(worker, event_loop)
        self.asyncio_task = event_loop.create_task(self._progress_task())
        self.worker.init_blocking_progress_mode()

    async def _progress_task(self):
        """This helper function maintains a UCX progress loop."""
        while True:
            worker = self.worker
            if worker is None:
                return
            worker.progress()
            # Give other co-routines a chance to run.
            await asyncio.sleep(0)
