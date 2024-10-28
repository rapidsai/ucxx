# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


import asyncio
import socket
import time
import weakref
from functools import partial

from ucxx._lib.libucxx import UCXWorker


class ProgressTask(object):
    def __init__(self, worker, event_loop):
        """Creates a task that keeps calling worker.progress()

        Notice, class and created task is careful not to hold a
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

        event_loop_close_original = self.event_loop.close

        def _event_loop_close(event_loop_close_original, *args, **kwargs):
            if not self.event_loop.is_closed() and self.asyncio_task is not None:
                try:
                    self.asyncio_task.cancel()
                    self.event_loop.run_until_complete(self.asyncio_task)
                except asyncio.exceptions.CancelledError:
                    pass
                finally:
                    self.asyncio_task = None
                    event_loop_close_original(*args, **kwargs)

        self.event_loop.close = partial(_event_loop_close, event_loop_close_original)

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
        worker.start_progress_thread(polling_mode=polling_mode, epoll_timeout=1)

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


class BlockingMode(ProgressTask):
    def __init__(
        self,
        worker: UCXWorker,
        event_loop: asyncio.AbstractEventLoop,
        progress_timeout: float = 1.0,
    ):
        """Progress the UCX worker in blocking mode.

        The blocking progress mode ensure the worker is progresses whenever the
        UCX worker reports an event on its epoll file descriptor. In certain
        circumstances the epoll file descriptor may not present an event, thus
        the `progress_timeout` will ensure the UCX worker is progressed to
        prevent a potential deadlock.

        Parameters
        ----------
        worker: UCXWorker
            Worker object from the UCXX Cython API to progress.
        event_loop: asyncio.AbstractEventLoop
            Asynchronous event loop where to schedule async tasks.
        progress_timeout: float
            The timeout to sleep until calling checking again whether the worker should
            be progressed.
        """
        super().__init__(worker, event_loop)
        self._progress_timeout = progress_timeout

        self.worker.init_blocking_progress_mode()

        # Creating a job that is ready straight away but with low priority.
        # Calling `await self.event_loop.sock_recv(self.rsock, 1)` will
        # return when all non-IO tasks are finished.
        # See <https://stackoverflow.com/a/48491563>.
        self.rsock, wsock = socket.socketpair()
        self.rsock.setblocking(0)
        wsock.setblocking(0)
        wsock.close()

        epoll_fd = self.worker.epoll_file_descriptor

        # Bind an asyncio reader to a UCX epoll file descriptor
        event_loop.add_reader(epoll_fd, self._fd_reader_callback)

        # Remove the reader and close socket on finalization
        weakref.finalize(self, event_loop.remove_reader, epoll_fd)
        weakref.finalize(self, self.rsock.close)

        self.blocking_asyncio_task = None
        self.last_progress_time = time.monotonic() - self._progress_timeout
        self.asyncio_task = event_loop.create_task(self._progress_with_timeout())

    def _fd_reader_callback(self):
        """Schedule new progress task upon worker event.

        Schedule new progress task when a new event occurs in the worker's epoll file
        descriptor.
        """
        self.worker.progress()

        # Notice, we can safely overwrite `self.blocking_asyncio_task`
        # since previous arm task is finished by now.
        assert self.blocking_asyncio_task is None or self.blocking_asyncio_task.done()
        self.blocking_asyncio_task = self.event_loop.create_task(self._arm_worker())

    async def _arm_worker(self):
        """Progress the worker and rearm.

        Progress and rearm the worker to watch for new events on its epoll file
        descriptor.
        """
        # When arming the worker, the following must be true:
        #  - No more progress in UCX (see doc of ucp_worker_arm())
        #  - All asyncio tasks that aren't waiting on UCX must be executed
        #    so that the asyncio's next state is epoll wait.
        #    See <https://github.com/rapidsai/ucx-py/issues/413>
        while True:
            self.last_progress_time = time.monotonic()
            self.worker.progress()

            # This IO task returns when all non-IO tasks are finished.
            # Notice, we do NOT hold a reference to `worker` while waiting.
            await self.event_loop.sock_recv(self.rsock, 1)

            if self.worker.arm():
                # At this point we know that asyncio's next state is
                # epoll wait.
                break

    async def _progress_with_timeout(self):
        """Protect worker from never progressing again.

        To ensure the worker progresses if no events are raised and the asyncio loop
        getting stuck we must ensure the worker is progressed every so often. This
        method ensures the worker is progressed independent of what the epoll file
        descriptor does if longer than `self._progress_timeout` has elapsed since
        last check, thus preventing a deadlock.
        """
        while True:
            worker = self.worker
            if worker is None:
                return
            if time.monotonic() > self.last_progress_time + self._progress_timeout:
                self.last_progress_time = time.monotonic()

                # Cancel `_arm_worker` task if available. `loop.sock_recv` does not
                # seem to respect timeout with `asyncio.wait_for`, thus we cancel
                # it here instead. It will get recreated after a new event on
                # `worker.epoll_file_descriptor`.
                if self.blocking_asyncio_task is not None:
                    self.blocking_asyncio_task.cancel()
                    try:
                        await self.blocking_asyncio_task
                    except asyncio.exceptions.CancelledError:
                        pass

                worker.progress()
            # Give other co-routines a chance to run.
            await asyncio.sleep(self._progress_timeout)
