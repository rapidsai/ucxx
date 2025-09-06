# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
from queue import Queue
from time import monotonic, sleep

import ucxx
import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx._lib_async.utils import get_event_loop
from ucxx.benchmarks.backends.base import BaseClient, BaseServer
from ucxx.benchmarks.utils import get_allocator
from ucxx.utils import print_key_value

WireupMessage = bytearray(b"wireup")


def _create_cuda_context(device):
    import numba.cuda

    numba.cuda.current_context(0)


def _transfer_wireup(ep, server):
    import numpy as np

    # Using bytearray currently segfaults
    # TODO: fix
    # message = bytearray(b"wireup")

    message = np.array([1], dtype="u8")
    if server:
        message = Array(message)
        return [
            ep.tag_recv(message, tag=ucx_api.UCXXTag(1)),
            ep.tag_send(message, tag=ucx_api.UCXXTag(0)),
        ]
    else:
        message = Array(np.zeros_like(message))
        return [
            ep.tag_send(message, tag=ucx_api.UCXXTag(1)),
            ep.tag_recv(message, tag=ucx_api.UCXXTag(0)),
        ]


async def _wait_requests_async(worker, requests):
    import asyncio

    await asyncio.gather(*[r.wait_yield() for r in requests])


def _wait_requests(worker, progress_mode, requests):
    while not all([r.completed for r in requests]):
        if progress_mode == "blocking":
            worker.progress_worker_event()
        if progress_mode == "polling":
            worker.progress()


def register_am_allocators(args: Namespace, worker: ucx_api.UCXWorker):
    """
    Register Active Message allocator in worker to correct memory type if the
    benchmark is set to use the Active Message API.

    Parameters
    ----------
    args
        Parsed command-line arguments that will be used as parameters during to
        determine whether the caller is using the Active Message API and what
        memory type.
    worker
        UCX-Py core Worker object where to register the allocator.
    """
    if not args.enable_am:
        return

    import numpy as np

    worker.register_am_allocator(
        lambda n: np.empty(n, dtype=np.uint8), ucx_api.AllocatorType.HOST
    )

    if args.object_type == "cupy":
        import cupy as cp

        worker.register_am_allocator(
            lambda n: cp.empty(n, dtype=cp.uint8), ucx_api.AllocatorType.CUDA
        )
    elif args.object_type == "rmm":
        import rmm

        worker.register_am_allocator(
            lambda n: rmm.DeviceBuffer(size=n), ucx_api.AllocatorType.CUDA
        )


class UCXPyCoreServer(BaseServer):
    has_cuda_support = True

    def __init__(
        self,
        args: Namespace,
        queue: Queue,
    ):
        self.args = args
        self.queue = queue

    def run(self):
        self.ep = None

        ctx = ucx_api.UCXContext(
            feature_flags=(
                ucx_api.Feature.AM if self.args.enable_am else ucx_api.Feature.TAG,
                ucx_api.Feature.WAKEUP,
            )
        )
        worker = ucx_api.UCXWorker(ctx)

        xp = get_allocator(
            self.args.object_type,
            self.args.rmm_init_pool_size,
            self.args.rmm_managed_memory,
        )

        register_am_allocators(self.args, worker)

        if self.args.progress_mode.startswith("thread"):
            worker.set_progress_thread_start_callback(
                _create_cuda_context, cb_args=(self.args.server_dev,)
            )
            polling_mode = self.args.progress_mode == "thread-polling"
            worker.start_progress_thread(polling_mode=polling_mode)
        else:
            worker.init_blocking_progress_mode()

        # A reference to listener's endpoint is stored to prevent it from going
        # out of scope immediately after the listener callback terminates.
        global ep
        ep = None

        def _listener_handler(conn_request):
            global ep
            ep = listener.create_endpoint_from_conn_request(
                conn_request, endpoint_error_handling=self.args.error_handling
            )

        listener = ucx_api.UCXListener.create(
            worker=worker, port=self.args.port or 0, cb_func=_listener_handler
        )
        self.queue.put(listener.port)

        # Without this, q.get() in main() may sometimes hang indefinitely.
        # TODO: find root cause and fix.
        sleep(0.1)

        while ep is None:
            if self.args.progress_mode == "blocking":
                worker.progress_worker_event()
            elif self.args.progress_mode == "polling":
                worker.progress()

        # Wireup before starting to transfer data
        wireup_requests = _transfer_wireup(ep, server=True)
        _wait_requests(worker, self.args.progress_mode, wireup_requests)

        async def _transfer():
            if self.args.reuse_alloc:
                recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

            for i in range(self.args.n_iter + self.args.n_warmup_iter):
                if not self.args.reuse_alloc:
                    recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

                requests = [
                    ep.tag_recv(recv_msg, tag=ucx_api.UCXXTag(1)),
                    ep.tag_send(recv_msg, tag=ucx_api.UCXXTag(0)),
                ]

                if self.args.asyncio_wait:
                    await _wait_requests_async(worker, requests)
                else:
                    _wait_requests(worker, self.args.progress_mode, requests)

                    # Check all requests completed successfully
                    for r in requests:
                        r.check_error()

        loop = get_event_loop()
        loop.run_until_complete(_transfer())


class UCXPyCoreClient(BaseClient):
    has_cuda_support = True

    def __init__(
        self,
        args: Namespace,
        queue: Queue,
        server_address: str,
        port: int,
    ):
        self.args = args
        self.queue = queue
        self.server_address = server_address
        self.port = port

    def run(self):
        ctx = ucx_api.UCXContext(
            feature_flags=(
                ucx_api.Feature.AM
                if self.args.enable_am is True
                else ucx_api.Feature.TAG,
                ucx_api.Feature.WAKEUP,
            )
        )
        worker = ucx_api.UCXWorker(ctx)

        xp = get_allocator(
            self.args.object_type,
            self.args.rmm_init_pool_size,
            self.args.rmm_managed_memory,
        )
        register_am_allocators(self.args, worker)
        send_msg = Array(xp.arange(self.args.n_bytes, dtype="u1"))

        if self.args.progress_mode.startswith("thread"):
            worker.set_progress_thread_start_callback(
                _create_cuda_context, cb_args=(self.args.client_dev,)
            )
            polling_mode = self.args.progress_mode == "thread-polling"
            worker.start_progress_thread(polling_mode=polling_mode)
        else:
            worker.init_blocking_progress_mode()

        ep = ucx_api.UCXEndpoint.create(
            worker,
            self.server_address,
            self.port,
            endpoint_error_handling=self.args.error_handling,
        )

        # Wireup before starting to transfer data
        wireup_requests = _transfer_wireup(ep, server=False)
        _wait_requests(worker, self.args.progress_mode, wireup_requests)

        times = []
        contention_metric = None

        async def _transfer():
            nonlocal contention_metric
            if self.args.reuse_alloc:
                recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

            if self.args.cuda_profile:
                xp.cuda.profiler.start()
            if self.args.report_gil_contention:
                from gilknocker import KnockKnock

                # Use smallest polling interval
                # possible to ensure, contention will always
                # be zero for small messages otherwise
                # and inconsistent for large messages.
                knocker = KnockKnock(polling_interval_micros=1)
                knocker.start()

            for i in range(self.args.n_iter + self.args.n_warmup_iter):
                start = monotonic()

                if not self.args.reuse_alloc:
                    recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

                requests = [
                    ep.tag_send(send_msg, tag=ucx_api.UCXXTag(1)),
                    ep.tag_recv(recv_msg, tag=ucx_api.UCXXTag(0)),
                ]

                if self.args.asyncio_wait:
                    await _wait_requests_async(worker, requests)
                else:
                    _wait_requests(worker, self.args.progress_mode, requests)

                    # Check all requests completed successfully
                    for r in requests:
                        r.check_error()

                stop = monotonic()
                if i >= self.args.n_warmup_iter:
                    times.append(stop - start)

            if self.args.report_gil_contention:
                knocker.stop()
            contention_metric = knocker.contention_metric
            if self.args.cuda_profile:
                xp.cuda.profiler.stop()

        loop = get_event_loop()
        loop.run_until_complete(_transfer())

        self.queue.put(times)
        if self.args.report_gil_contention:
            self.queue.put(contention_metric)

    def print_backend_specific_config(self):
        delay_progress_str = (
            f"True ({self.args.max_outstanding})"
            if self.args.delay_progress is True
            else "False"
        )

        print_key_value(
            key="Transfer API", value=f"{'AM' if self.args.enable_am else 'TAG'}"
        )
        print_key_value(key="Progress mode", value=f"{self.args.progress_mode}")
        print_key_value(key="Asyncio wait", value=f"{self.args.asyncio_wait}")
        print_key_value(key="Delay progress", value=f"{delay_progress_str}")
        print_key_value(key="UCX_TLS", value=f"{ucxx.get_config()['TLS']}")
        print_key_value(
            key="UCX_NET_DEVICES", value=f"{ucxx.get_config()['NET_DEVICES']}"
        )
