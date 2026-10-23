# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import subprocess
import sys
import textwrap

import pytest


def test_future_task_example_snapshot_includes_suspended_tasks():
    from ucxx.examples.python_future_task_example import _asyncio_task_snapshot

    async def wait_forever():
        await asyncio.Event().wait()

    async def snapshot_while_task_is_suspended():
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(wait_forever(), name="future-example-test-task")
        await asyncio.sleep(0)
        try:
            return _asyncio_task_snapshot(loop)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    snapshot = asyncio.run(snapshot_while_task_is_suspended())
    assert "future-example-test-task" in snapshot
    assert "wait_forever" in snapshot


def test_future_task_shutdown_does_not_deadlock_waiting_for_gil():
    program = textwrap.dedent(
        """
        import asyncio
        from ucxx.examples.python_future_task_app import PythonFutureTaskApplication

        async def main():
            app = PythonFutureTaskApplication(asyncio.get_running_loop())
            future = app.submit_until_close(id=17)
            print("waiting for progress thread to accept task", flush=True)
            app.wait_until_task_accepted()
            print("progress thread accepted task", flush=True)
            print("destroying application", flush=True)
            del app
            print("application destroyed", flush=True)
            print("awaiting future", flush=True)
            assert await future == 17
            print("future completed after application teardown", flush=True)

        asyncio.run(main())
        """
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "future task shutdown deadlocked after the progress thread accepted "
            "the task\n"
            f"stdout: {exc.stdout}\nstderr: {exc.stderr}",
            pytrace=False,
        )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "destroying application" in result.stdout
    assert "application destroyed" in result.stdout
    assert "awaiting future" in result.stdout
    assert "future completed after application teardown" in result.stdout
