# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import textwrap

import pytest


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
            del app
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
    assert "future completed after application teardown" in result.stdout
