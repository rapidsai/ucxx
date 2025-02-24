# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import select
from contextlib import contextmanager
from time import sleep, time

import pytest

import dask
from distributed import Client
from distributed.utils import get_ip, open_port
from distributed.utils_test import popen

from distributed_ucxx.ucxx import _prepare_ucx_config
from distributed_ucxx.utils_test import gen_test

try:
    HOST = get_ip()
except Exception:
    HOST = "127.0.0.1"

ucxx = pytest.importorskip("ucxx")
rmm = pytest.importorskip("rmm")


@gen_test()
async def test_ucx_config(ucxx_loop, cleanup):
    ucx = {
        "nvlink": True,
        "infiniband": True,
        "rdmacm": False,
        "tcp": True,
        "cuda-copy": True,
    }

    with dask.config.set({"distributed.comm.ucx": ucx}):
        ucx_config, ucx_environment = _prepare_ucx_config()
        assert ucx_config == {
            "TLS": "rc,tcp,cuda_copy,cuda_ipc",
            "SOCKADDR_TLS_PRIORITY": "tcp",
        }
        assert ucx_environment == {}

    ucx = {
        "nvlink": False,
        "infiniband": True,
        "rdmacm": False,
        "tcp": True,
        "cuda-copy": False,
    }

    with dask.config.set({"distributed.comm.ucx": ucx}):
        ucx_config, ucx_environment = _prepare_ucx_config()
        assert ucx_config == {"TLS": "rc,tcp", "SOCKADDR_TLS_PRIORITY": "tcp"}
        assert ucx_environment == {}

    ucx = {
        "nvlink": False,
        "infiniband": True,
        "rdmacm": True,
        "tcp": True,
        "cuda-copy": True,
    }

    with dask.config.set({"distributed.comm.ucx": ucx}):
        ucx_config, ucx_environment = _prepare_ucx_config()
        assert ucx_config == {
            "TLS": "rc,tcp,cuda_copy",
            "SOCKADDR_TLS_PRIORITY": "rdmacm",
        }
        assert ucx_environment == {}

    ucx = {
        "nvlink": None,
        "infiniband": None,
        "rdmacm": None,
        "tcp": None,
        "cuda-copy": None,
    }

    with dask.config.set({"distributed.comm.ucx": ucx}):
        ucx_config, ucx_environment = _prepare_ucx_config()
        assert ucx_config == {}
        assert ucx_environment == {}

    ucx = {
        "nvlink": False,
        "infiniband": True,
        "rdmacm": True,
        "tcp": True,
        "cuda-copy": True,
    }

    with dask.config.set(
        {
            "distributed.comm.ucx": ucx,
            "distributed.comm.ucx.environment": {
                "tls": "all",
                "memtrack-dest": "stdout",
            },
        }
    ):
        ucx_config, ucx_environment = _prepare_ucx_config()
        assert ucx_config == {
            "TLS": "rc,tcp,cuda_copy",
            "SOCKADDR_TLS_PRIORITY": "rdmacm",
        }
        assert ucx_environment == {"UCX_MEMTRACK_DEST": "stdout"}


@contextmanager
def start_dask_scheduler(env: list[str], max_attempts: int = 5, timeout: int = 10):
    """
    Start Dask scheduler in subprocess.

    Attempts to start a Dask scheduler in subprocess, if the port is not available
    retry on a different port up to a maximum of `max_attempts` attempts. The stdout
    and stderr of the process is read to determine whether the scheduler failed to
    bind to port or succeeded, and ensures no more than `timeout` seconds are awaited
    for between reads.

    Parameters
    ----------
    env: list[str]
        Environment variables to start scheduler process with.
    max_attempts: int
        Maximum attempts to try to open scheduler.
    timeout: int
        Time to wait while reading stdout/stderr of subprocess.
    """
    retry_count = 0

    port = open_port()
    while retry_count < max_attempts:
        with popen(
            [
                "dask",
                "scheduler",
                "--no-dashboard",
                "--protocol",
                "ucxx",
                "--port",
                str(port),
            ],
            env=env,
            capture_output=True,  # Capture stdout and stderr
        ) as scheduler_process:
            # Check if the scheduler process started successfully by streaming output
            try:
                start_time = time()
                while True:
                    if time() - start_time > timeout:
                        raise TimeoutError("Timeout while waiting for scheduler output")

                    # Use select to wait for data with a timeout
                    ready, _, _ = select.select(
                        [scheduler_process.stdout], [], [], timeout
                    )
                    if scheduler_process.stdout in ready:
                        line = scheduler_process.stdout.readline()
                        if not line:
                            break  # End of output
                        print(
                            line.decode(), end=""
                        )  # Since capture_output=True, print the line here
                        if b"Scheduler at:" in line:
                            # Scheduler is now listening
                            break
                        elif b"UCXXBusyError" in line:
                            raise Exception(
                                "UCXXBusyError detected in scheduler output"
                            )
            except Exception:
                retry_count += 1
                port += 1
                continue

            yield scheduler_process, port

            return
    else:
        pytest.fail(f"Failed to start dask scheduler after {max_attempts} attempts.")


@pytest.mark.skipif(
    int(os.environ.get("UCXPY_ENABLE_PYTHON_FUTURE", "0")) != 0,
    reason="Workers running without a `Nanny` can't be closed properly",
)
def test_ucx_config_w_env_var(ucxx_loop, cleanup, loop):
    env = os.environ.copy()
    env["DASK_DISTRIBUTED__RMM__POOL_SIZE"] = "1000.00 MB"

    with start_dask_scheduler(env=env) as scheduler_process_port:
        scheduler_process, scheduler_port = scheduler_process_port
        sched_addr = f"ucxx://127.0.0.1:{scheduler_port}"
        print(f"{sched_addr=}", flush=True)

        with popen(
            [
                "dask",
                "worker",
                sched_addr,
                "--host",
                "127.0.0.1",
                "--no-dashboard",
                "--protocol",
                "ucxx",
                "--no-nanny",
            ],
            env=env,
        ):
            with Client(sched_addr, loop=loop, timeout=60) as c:
                while not c.scheduler_info()["workers"]:
                    sleep(0.1)

                # Check for RMM pool resource type
                rmm_resource = c.run_on_scheduler(
                    rmm.mr.get_current_device_resource_type
                )
                assert rmm_resource == rmm.mr.PoolMemoryResource

                rmm_resource_workers = c.run(rmm.mr.get_current_device_resource_type)
                for v in rmm_resource_workers.values():
                    assert v == rmm.mr.PoolMemoryResource
