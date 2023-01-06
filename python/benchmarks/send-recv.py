"""
Benchmark send receive on one machine:
UCX_TLS=tcp,cuda_copy,cuda_ipc python send-recv.py \
        --server-dev 2 --client-dev 1 --object_type rmm \
        --reuse-alloc --n-bytes 1GB


Benchmark send receive on two machines (IB testing):
# server process
UCX_MAX_RNDV_RAILS=1 UCX_TLS=tcp,cuda_copy,rc python send-recv.py \
        --server-dev 0 --client-dev 5 --object_type rmm --reuse-alloc \
        --n-bytes 1GB --server-only --port 13337 --n-iter 100

# client process
UCX_MAX_RNDV_RAILS=1 UCX_TLS=tcp,cuda_copy,rc python send-recv.py \
        --server-dev 0 --client-dev 5 --object_type rmm --reuse-alloc \
        --n-bytes 1GB --client-only --server-address SERVER_IP --port 13337 \
        --n-iter 100
"""
import argparse
import asyncio
import multiprocessing as mp
import os
import time
from time import perf_counter as clock

from dask.utils import format_bytes, parse_bytes

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx._lib_async.utils import get_event_loop

mp = mp.get_context("spawn")


def print_separator(separator="-", length=80):
    print(separator * length)


def print_key_value(key, value, key_length=25):
    print(f"{key: <{key_length}} | {value}")


def _create_cuda_context(device):
    import numba.cuda

    numba.cuda.current_context(device)


def _transfer_wireup(ep, server):
    import numpy as np

    # Using bytearray currently segfaults
    # TODO: fix
    # message = bytearray(b"wireup")

    message = np.array([1], dtype="u8")
    if server:
        message = Array(message)
        return [
            ep.tag_recv(message, tag=0),
            ep.tag_send(message, tag=0),
        ]
    else:
        message = Array(np.zeros_like(message))
        return [
            ep.tag_send(message, tag=0),
            ep.tag_recv(message, tag=0),
        ]


async def _wait_requests_async(worker, requests):
    await asyncio.gather(*[r.wait_yield() for r in requests])


def _wait_requests(worker, progress_mode, requests):
    while not all([r.is_completed() for r in requests]):
        if progress_mode == "blocking":
            worker.progress_worker_event()


def server(queue, args):
    if len(args.server_cpu_affinity) > 0:
        os.sched_setaffinity(0, args.server_cpu_affinity)

    if args.object_type == "numpy":
        import numpy as xp
    elif args.object_type == "cupy":
        import cupy as xp

        xp.cuda.runtime.setDevice(args.server_dev)
    else:
        import cupy as xp

        import rmm

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=False,
            initial_pool_size=args.rmm_init_pool_size,
            devices=[args.server_dev],
        )
        xp.cuda.runtime.setDevice(args.server_dev)
        xp.cuda.set_allocator(rmm.rmm_cupy_allocator)

    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx, enable_delayed_submission=args.delayed_submission)

    if args.progress_mode == "thread":
        worker.set_progress_thread_start_callback(
            _create_cuda_context, cb_args=(args.server_dev,)
        )
        worker.start_progress_thread()
    else:
        worker.init_blocking_progress_mode()

    # A reference to listener's endpoint is stored to prevent it from going
    # out of scope immediately after the listener callback terminates.
    global ep
    ep = None

    def _listener_handler(conn_request):
        global ep
        ep = listener.create_endpoint_from_conn_request(conn_request, True)

    listener = ucx_api.UCXListener.create(
        worker=worker, port=args.port or 0, cb_func=_listener_handler
    )
    queue.put(listener.port)

    # Without this, q.get() in main() may sometimes hang indefinitely.
    # TODO: find root cause and fix.
    time.sleep(0.1)

    while ep is None:
        if args.progress_mode == "blocking":
            worker.progress_worker_event()

    # Wireup before starting to transfer data
    wireup_requests = _transfer_wireup(ep, server=True)
    _wait_requests(worker, args.progress_mode, wireup_requests)

    async def _transfer():
        if args.reuse_alloc:
            recv_msg = Array(xp.zeros(args.n_bytes, dtype="u1"))

        for i in range(args.n_iter + args.n_warmup_iter):
            if not args.reuse_alloc:
                recv_msg = Array(xp.zeros(args.n_bytes, dtype="u1"))

            requests = [
                ep.tag_recv(recv_msg, tag=0),
                ep.tag_send(recv_msg, tag=0),
            ]

            if args.asyncio_wait:
                await _wait_requests_async(worker, requests)
            else:
                _wait_requests(worker, args.progress_mode, requests)

                # Check all requests completed successfully
                for r in requests:
                    r.check_error()

    loop = get_event_loop()
    loop.run_until_complete(_transfer())


def client(port, server_address, args):
    if len(args.client_cpu_affinity) > 0:
        os.sched_setaffinity(0, args.client_cpu_affinity)

    import numpy as np

    if args.object_type == "numpy":
        import numpy as xp
    elif args.object_type == "cupy":
        import cupy as xp

        xp.cuda.runtime.setDevice(args.client_dev)
    else:
        import cupy as xp

        import rmm

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=False,
            initial_pool_size=args.rmm_init_pool_size,
            devices=[args.client_dev],
        )
        xp.cuda.runtime.setDevice(args.client_dev)
        xp.cuda.set_allocator(rmm.rmm_cupy_allocator)

    send_msg = Array(xp.arange(args.n_bytes, dtype="u1"))

    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx, enable_delayed_submission=args.delayed_submission)

    if args.progress_mode == "thread":
        worker.set_progress_thread_start_callback(
            _create_cuda_context, cb_args=(args.client_dev,)
        )
        worker.start_progress_thread()
    else:
        worker.init_blocking_progress_mode()

    ep = ucx_api.UCXEndpoint.create(
        worker,
        server_address,
        port,
        endpoint_error_handling=True,
    )

    # Wireup before starting to transfer data
    wireup_requests = _transfer_wireup(ep, server=False)
    _wait_requests(worker, args.progress_mode, wireup_requests)

    times = []

    async def _transfer():
        if args.reuse_alloc:
            recv_msg = Array(xp.zeros(args.n_bytes, dtype="u1"))

        if args.cuda_profile:
            xp.cuda.profiler.start()

        for i in range(args.n_iter + args.n_warmup_iter):
            start = clock()

            if not args.reuse_alloc:
                recv_msg = Array(xp.zeros(args.n_bytes, dtype="u1"))

            requests = [
                ep.tag_send(send_msg, tag=0),
                ep.tag_recv(recv_msg, tag=0),
            ]

            if args.asyncio_wait:
                await _wait_requests_async(worker, requests)
            else:
                _wait_requests(worker, args.progress_mode, requests)

                # Check all requests completed successfully
                for r in requests:
                    r.check_error()

            stop = clock()
            if i >= args.n_warmup_iter:
                times.append(stop - start)

        if args.cuda_profile:
            xp.cuda.profiler.stop()

    loop = get_event_loop()
    loop.run_until_complete(_transfer())

    assert len(times) == args.n_iter
    bw_avg = format_bytes(2 * args.n_iter * args.n_bytes / sum(times))
    bw_med = format_bytes(2 * args.n_bytes / np.median(times))
    lat_avg = int(sum(times) * 1e9 / (2 * args.n_iter))
    lat_med = int(np.median(times) * 1e9 / 2)

    print("Roundtrip benchmark")
    print_separator(separator="=")
    print_key_value(key="Iterations", value=f"{args.n_iter}")
    print_key_value(key="Bytes", value=f"{format_bytes(args.n_bytes)}")
    print_key_value(key="Object type", value=f"{args.object_type}")
    print_key_value(key="Reuse allocation", value=f"{args.reuse_alloc}")
    print_key_value(key="Progress mode", value=f"{args.progress_mode}")
    print_key_value(key="Asyncio wait", value=f"{args.asyncio_wait}")
    print_key_value(key="UCX_TLS", value=f"{ctx.get_config()['TLS']}")
    print_key_value(key="UCX_NET_DEVICES", value=f"{ctx.get_config()['NET_DEVICES']}")
    print_separator(separator="=")
    if args.object_type == "numpy":
        print_key_value(key="Device(s)", value="CPU-only")
        s_aff = (
            args.server_cpu_affinity
            if len(args.server_cpu_affinity) > 0
            else "affinity not set"
        )
        c_aff = (
            args.client_cpu_affinity
            if len(args.client_cpu_affinity) > 0
            else "affinity not set"
        )
        print_key_value(key="Server CPU", value=f"{s_aff}")
        print_key_value(key="Client CPU", value=f"{c_aff}")
    else:
        print_key_value(key="Device(s)", value=f"{args.server_dev}, {args.client_dev}")
    print_separator(separator="=")
    print_key_value("Bandwidth (average)", value=f"{bw_avg}/s")
    print_key_value("Bandwidth (median)", value=f"{bw_med}/s")
    print_key_value("Latency (average)", value=f"{lat_avg} ns")
    print_key_value("Latency (median)", value=f"{lat_med} ns")
    if not args.no_detailed_report:
        print_separator(separator="=")
        print_key_value(key="Iterations", value="Bandwidth, Latency")
        print_separator(separator="-")
        for i, t in enumerate(times):
            ts = format_bytes(2 * args.n_bytes / t)
            lat = int(t * 1e9 / 2)
            print_key_value(key=i, value=f"{ts}/s, {lat}ns")


def _parse_cpu_affinity(affinity_str):
    assert isinstance(affinity_str, str)
    if len(affinity_str) == 0:
        return tuple()
    try:
        return (int(affinity_str),)
    except ValueError:
        return tuple(affinity_str.split(","))


def parse_args():
    parser = argparse.ArgumentParser(description="Roundtrip benchmark")
    parser.add_argument(
        "-n",
        "--n-bytes",
        metavar="BYTES",
        default="10 Mb",
        type=parse_bytes,
        help="Message size. Default '10 Mb'.",
    )
    parser.add_argument(
        "--n-iter",
        metavar="N",
        default=10,
        type=int,
        help="Number of send / recv iterations (default 10).",
    )
    parser.add_argument(
        "--n-warmup-iter",
        default=10,
        type=int,
        help="Number of send / recv warmup iterations (default 10).",
    )
    parser.add_argument(
        "-b",
        "--server-cpu-affinity",
        metavar="N",
        default=str(),
        type=str,
        help="CPU affinity for server process (default '': not set).",
    )
    parser.add_argument(
        "-c",
        "--client-cpu-affinity",
        metavar="N",
        default=str(),
        type=str,
        help="CPU affinity for client process (default '': not set).",
    )
    parser.add_argument(
        "-o",
        "--object_type",
        default="numpy",
        choices=["numpy", "cupy", "rmm"],
        help="In-memory array type.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Whether to print timings per iteration.",
    )
    parser.add_argument(
        "-s",
        "--server-address",
        metavar="ip",
        # default=ucp.get_address(),
        default="127.0.0.1",
        type=str,
        help="Server address (default `ucp.get_address()`).",
    )
    parser.add_argument(
        "-d",
        "--server-dev",
        metavar="N",
        default=0,
        type=int,
        help="GPU device on server (default 0).",
    )
    parser.add_argument(
        "-e",
        "--client-dev",
        metavar="N",
        default=0,
        type=int,
        help="GPU device on client (default 0).",
    )
    parser.add_argument(
        "--reuse-alloc",
        default=False,
        action="store_true",
        help="Reuse memory allocations between communication.",
    )
    parser.add_argument(
        "--cuda-profile",
        default=False,
        action="store_true",
        help="Setting CUDA profiler.start()/stop() around send/recv "
        "typically used with `nvprof --profile-from-start off "
        "--profile-child-processes`",
    )
    parser.add_argument(
        "--rmm-init-pool-size",
        metavar="BYTES",
        default=None,
        type=int,
        help="Initial RMM pool size (default  1/2 total GPU memory)",
    )
    parser.add_argument(
        "--server-only",
        default=False,
        action="store_true",
        help="Start up only a server process (to be used with --client).",
    )
    parser.add_argument(
        "--client-only",
        default=False,
        action="store_true",
        help="Connect to solitary server process (to be user with --server-only)",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=None,
        help="The port the server will bind to, if not specified, UCX will bind "
        "to a random port. Must be specified when --client-only is used.",
        type=int,
    )
    parser.add_argument(
        "--progress-mode",
        default="thread",
        help="Progress for the UCP worker. Valid options are: "
        "'thread' (default) and 'blocking'.",
        type=str,
    )
    parser.add_argument(
        "--asyncio-wait",
        default=False,
        action="store_true",
        help="Wait for transfer requests with Python's asyncio, requires"
        "`--progress-mode=thread`. (Default: disabled)",
    )
    parser.add_argument(
        "--delayed-submission",
        default=False,
        action="store_true",
        help="Enable delayed submission. (Default: disabled)",
    )
    parser.add_argument(
        "--no-detailed-report",
        default=False,
        action="store_true",
        help="Disable detailed report per iteration.",
    )

    args = parser.parse_args()
    if args.cuda_profile and args.object_type == "numpy":
        raise RuntimeError(
            "`--cuda-profile` requires `--object_type=cupy` or `--object_type=rmm`"
        )
    if args.progress_mode != "blocking" and args.progress_mode != "thread":
        raise RuntimeError(f"Invalid `--progress-mode`: '{args.progress_mode}'")
    if args.asyncio_wait and args.progress_mode != "thread":
        raise RuntimeError("`--asyncio-wait` requires `--progress-mode=thread`")
    args.server_cpu_affinity = _parse_cpu_affinity(args.server_cpu_affinity)
    args.client_cpu_affinity = _parse_cpu_affinity(args.client_cpu_affinity)
    return args


def main():
    args = parse_args()
    server_address = args.server_address

    # if you are the server, only start the `server process`
    # if you are the client, only start the `client process`
    # otherwise, start everything

    if not args.client_only:
        # server process
        q1 = mp.Queue()
        p1 = mp.Process(target=server, args=(q1, args))
        p1.start()
        port = q1.get()
        print(f"Server Running at {server_address}:{port}")
    else:
        port = args.port

    if not args.server_only or args.client_only:
        # client process
        p2 = mp.Process(target=client, args=(port, server_address, args))
        p2.start()
        p2.join()
        assert not p2.exitcode

    else:
        p1.join()
        assert not p1.exitcode


if __name__ == "__main__":
    main()
