# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import contextlib
from random import random
from typing import List

from ucxx.examples.python_future_task_app import PythonFutureTaskApplication


def _asyncio_task_snapshot(loop):
    """Return task states and their suspended coroutine await chains."""
    snapshots = []
    for task in sorted(asyncio.all_tasks(loop), key=lambda task: task.get_name()):
        await_chain = []
        awaited = task.get_coro()
        seen = set()
        while awaited is not None and len(await_chain) < 12 and id(awaited) not in seen:
            seen.add(id(awaited))
            frame = (
                getattr(awaited, "cr_frame", None)
                or getattr(awaited, "gi_frame", None)
                or getattr(awaited, "ag_frame", None)
            )
            code = (
                getattr(awaited, "cr_code", None)
                or getattr(awaited, "gi_code", None)
                or getattr(awaited, "ag_code", None)
            )
            if code is None:
                await_chain.append(type(awaited).__name__)
            else:
                line = None if frame is None else frame.f_lineno
                await_chain.append(f"{code.co_name} ({code.co_filename}:{line})")
            awaited = (
                getattr(awaited, "cr_await", None)
                or getattr(awaited, "gi_yieldfrom", None)
                or getattr(awaited, "ag_await", None)
            )
        snapshots.append(
            {
                "name": task.get_name(),
                "done": task.done(),
                "cancelled": task.cancelled(),
                "await_chain": await_chain,
            }
        )
    return f"asyncio_tasks={snapshots}"


async def _dump_task_snapshot_periodically(loop):
    while True:
        await asyncio.sleep(10)
        print(
            f"Python future example watchdog: {_asyncio_task_snapshot(loop)}",
            flush=True,
        )


def submit_task(
    cpp_app: PythonFutureTaskApplication,
    num_tasks: int = 10,
    max_task_duration: float = 1.0,
) -> List[asyncio.Future]:
    return [
        cpp_app.submit(duration=random() * max_task_duration, id=t)
        for t in range(num_tasks)
    ]


async def main():
    loop = asyncio.get_running_loop()
    cpp_app = PythonFutureTaskApplication(loop)
    watchdog = asyncio.create_task(
        _dump_task_snapshot_periodically(loop),
        name="future-example-diagnostic-watchdog",
    )
    num_tasks = 10
    max_task_duration = 3.0

    try:
        tasks = submit_task(
            cpp_app=cpp_app, num_tasks=num_tasks, max_task_duration=max_task_duration
        )
        print(f"Tasks submitted count={len(tasks)}", flush=True)
        results = await asyncio.gather(*tasks)
        completed = sum(task.done() for task in tasks)
        print(
            f"Future results complete={completed}/{len(tasks)} results={results}",
            flush=True,
        )
        assert all(got == expected for got, expected in zip(results, range(num_tasks)))
    finally:
        print("Python future example: application shutdown starting", flush=True)
        del cpp_app
        print("Python future example: application shutdown complete", flush=True)
        watchdog.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
