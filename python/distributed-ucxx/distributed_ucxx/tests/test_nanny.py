# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio

from distributed import Nanny, Scheduler
from distributed.utils_test import gen_test
from distributed.worker import Worker
from distributed_ucxx.utils_test import _asyncio_task_snapshot


class KeyboardInterruptWorker(Worker):
    """A Worker that raises KeyboardInterrupt almost immediately"""

    async def heartbeat(self):
        def raise_err():
            raise KeyboardInterrupt()

        self.loop.add_callback(raise_err)


def test_nanny_lifecycle_snapshot_includes_asyncio_task_stacks():
    async def wait_forever():
        await asyncio.Event().wait()

    async def snapshot_while_task_is_suspended():
        task = asyncio.create_task(wait_forever(), name="lifecycle-diagnostic-probe")
        await asyncio.sleep(0)
        try:
            return _asyncio_task_snapshot()
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    details = asyncio.run(snapshot_while_task_is_suspended())
    assert "lifecycle-diagnostic-probe" in details
    assert "wait_forever" in details


# `ucxx_loop` checks threads after stopping UCXX's notifier during fixture teardown.
@gen_test(timeout=120, clean_kwargs={"threads": False})
async def test_nanny_closed_by_keyboard_interrupt(ucxx_loop):
    async with Scheduler(protocol="ucx", dashboard_address=":0") as s:
        async with Nanny(
            s.address, nthreads=1, worker_class=KeyboardInterruptWorker
        ) as n:
            await n.process.stopped.wait()
            # Check that the scheduler has been notified about the closed worker
            assert "remove-worker" in str(s.get_events())
