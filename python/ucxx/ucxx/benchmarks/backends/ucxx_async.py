# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import os
import sys
from argparse import Namespace
from collections import deque
from queue import Queue
from time import monotonic, time_ns

import ucxx
from ucxx._lib.arr import Array
from ucxx.benchmarks.backends.base import BaseClient, BaseServer
from ucxx.benchmarks.utils import get_allocator
from ucxx.utils import print_key_value


class _BenchmarkTrace:
    """Keep recent operation boundaries without logging timed iterations."""

    def __init__(self, role: str, capacity: int = 64):
        self.enabled = os.environ.get("UCXX_BENCH_TRACE") == "1"
        self.role = role
        self.records = deque(maxlen=capacity)

    def record(self, stage: str, iteration: int, endpoint: int):
        if self.enabled:
            self.records.append((time_ns(), stage, iteration, endpoint))

    def dump(self):
        if not self.enabled:
            return
        print(
            f"UCXX_BENCH_TRACE_BEGIN pid={os.getpid()} role={self.role} "
            f"count={len(self.records)}",
            file=sys.stderr,
        )
        for timestamp, stage, iteration, endpoint in self.records:
            print(
                f"UCXX_BENCH_TRACE time_ns={timestamp} pid={os.getpid()} "
                f"role={self.role} ep=0x{endpoint:x} iteration={iteration} "
                f"stage={stage}",
                file=sys.stderr,
            )
        print(f"UCXX_BENCH_TRACE_END pid={os.getpid()}", file=sys.stderr, flush=True)


def register_am_allocators(args: Namespace):
    """
    Register Active Message allocator in worker to correct memory type if the
    benchmark is set to use the Active Message API.

    Parameters
    ----------
    args
        Parsed command-line arguments that will be used as parameters during to
        determine whether the caller is using the Active Message API and what
        memory type.
    """
    if not args.enable_am:
        return

    import numpy as np

    ucxx.register_am_allocator(lambda n: np.empty(n, dtype=np.uint8), "host")

    if args.object_type == "cupy":
        import cupy as cp

        ucxx.register_am_allocator(lambda n: cp.empty(n, dtype=cp.uint8), "cuda")
    elif args.object_type == "rmm":
        import rmm

        ucxx.register_am_allocator(lambda n: rmm.DeviceBuffer(size=n), "cuda")


class UCXPyAsyncServer(BaseServer):
    has_cuda_support = True

    def __init__(
        self,
        args: Namespace,
        queue: Queue,
    ):
        self.args = args
        self.queue = queue

    async def run(self):
        ucxx.init(progress_mode=self.args.progress_mode)
        trace = _BenchmarkTrace("server")

        xp = get_allocator(
            self.args.object_type,
            self.args.rmm_init_pool_size,
            self.args.rmm_managed_memory,
        )

        register_am_allocators(self.args)

        async def server_handler(ep):
            i = -1
            try:
                if not self.args.enable_am:
                    if self.args.reuse_alloc and self.args.n_buffers == 1:
                        reuse_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

                for i in range(self.args.n_iter + self.args.n_warmup_iter):
                    if self.args.enable_am:
                        recv = await ep.am_recv()
                        await ep.am_send(recv)
                    else:
                        if self.args.n_buffers == 1:
                            msg = (
                                reuse_msg
                                if self.args.reuse_alloc
                                else xp.zeros(self.args.n_bytes, dtype="u1")
                            )
                            assert msg.nbytes == self.args.n_bytes

                            await ep.recv(msg)
                            await ep.send(msg)
                        else:
                            if trace.enabled:
                                trace.record("recv_multi-start", i, id(ep))
                            msgs = await ep.recv_multi()
                            if trace.enabled:
                                trace.record("recv_multi-done", i, id(ep))
                                trace.record("send_multi-start", i, id(ep))
                            await ep.send_multi(msgs)
                            if trace.enabled:
                                trace.record("send_multi-done", i, id(ep))
                trace.record("close-start", i, id(ep))
                await ep.close()
                trace.record("close-done", i, id(ep))
                lf.close()
            except BaseException as e:
                trace.record(f"error:{type(e).__name__}", i, id(ep))
                raise
            finally:
                trace.dump()

        lf = ucxx.create_listener(
            server_handler,
            port=self.args.port,
            endpoint_error_handling=self.args.error_handling,
        )
        self.queue.put(lf.port)

        while not lf.closed:
            await asyncio.sleep(0.5)

        ucxx.stop_notifier_thread()


class UCXPyAsyncClient(BaseClient):
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

    async def run(self):
        ucxx.init(progress_mode=self.args.progress_mode)
        trace = _BenchmarkTrace("client")

        xp = get_allocator(
            self.args.object_type,
            self.args.rmm_init_pool_size,
            self.args.rmm_managed_memory,
        )

        register_am_allocators(self.args)

        ep = await ucxx.create_endpoint(
            self.server_address,
            self.port,
            endpoint_error_handling=self.args.error_handling,
        )

        if self.args.enable_am:
            msg = xp.arange(self.args.n_bytes, dtype="u1")
        else:
            if self.args.reuse_alloc:
                reuse_msg_send = Array(xp.arange(self.args.n_bytes, dtype="u1"))

                if self.args.n_buffers == 1:
                    reuse_msg_recv = Array(xp.zeros(self.args.n_bytes, dtype="u1"))
                else:
                    reuse_msg_send = [reuse_msg_send] * self.args.n_buffers

        if self.args.cuda_profile:
            xp.cuda.profiler.start()
        if self.args.report_gil_contention:
            from gilknocker import KnockKnock

            # Use smallest polling interval possible to ensure, contention will always
            # be zero for small messages otherwise and inconsistent for large messages.
            knocker = KnockKnock(polling_interval_micros=1)
            knocker.start()

        times = []
        i = -1
        try:
            for i in range(self.args.n_iter + self.args.n_warmup_iter):
                start = monotonic()
                if self.args.enable_am:
                    await ep.am_send(msg)
                    await ep.am_recv()
                else:
                    if self.args.n_buffers == 1:
                        if self.args.reuse_alloc:
                            msg_send = reuse_msg_send
                            msg_recv = reuse_msg_recv
                        else:
                            msg_send = Array(xp.arange(self.args.n_bytes, dtype="u1"))
                            msg_recv = Array(xp.zeros(self.args.n_bytes, dtype="u1"))
                        await ep.send(msg_send)
                        await ep.recv(msg_recv)
                    else:
                        if self.args.reuse_alloc:
                            msg_send = reuse_msg_send
                        else:
                            msg_send = [
                                Array(xp.arange(self.args.n_bytes, dtype="u1"))
                                for i in range(self.args.n_buffers)
                            ]
                        if trace.enabled:
                            trace.record("send_multi-start", i, id(ep))
                        await ep.send_multi(msg_send)
                        if trace.enabled:
                            trace.record("send_multi-done", i, id(ep))
                            trace.record("recv_multi-start", i, id(ep))
                        msg_recv = await ep.recv_multi()
                        if trace.enabled:
                            trace.record("recv_multi-done", i, id(ep))
                stop = monotonic()
                if i >= self.args.n_warmup_iter:
                    times.append(stop - start)
        except BaseException as e:
            trace.record(f"error:{type(e).__name__}", i, id(ep))
            raise
        finally:
            trace.dump()

        if self.args.report_gil_contention:
            knocker.stop()
        if self.args.cuda_profile:
            xp.cuda.profiler.stop()

        self.queue.put(times)
        if self.args.report_gil_contention:
            self.queue.put(knocker.contention_metric)

        ucxx.stop_notifier_thread()

    def print_backend_specific_config(self):
        print_key_value(
            key="Transfer API", value=f"{'AM' if self.args.enable_am else 'TAG'}"
        )
        print_key_value(key="Progress mode", value=f"{self.args.progress_mode}")
        print_key_value(key="UCX_TLS", value=f"{ucxx.get_config()['TLS']}")
        print_key_value(
            key="UCX_NET_DEVICES", value=f"{ucxx.get_config()['NET_DEVICES']}"
        )
