import argparse
import asyncio

import numpy as np

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array


async def _wait_requests_async(worker, requests):
    await asyncio.gather(*[r.is_completed_async() for r in requests])


def _wait_requests(worker, progress_mode, requests):
    while not all([r.is_completed() for r in requests]):
        if progress_mode == "blocking":
            worker.progress_worker_event()


def parse_args():
    parser = argparse.ArgumentParser(description="Basic UCXX-Py Example")
    parser.add_argument(
        "--asyncio-wait",
        default=False,
        action="store_true",
        help="Wait for transfer requests with Python's asyncio, requires"
        "--threaded-progress. (Default: disabled)",
    )
    parser.add_argument(
        "-m",
        "--progress-mode",
        default="threaded",
        help="Progress mode for the UCP worker. Valid options are: "
        "'threaded' (default) and 'blocking'.",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--object-type",
        default="numpy",
        choices=["numpy", "rmm"],
        help="In-memory array type.",
        type=str,
    )
    parser.add_argument(
        "--multi-buffer-transfer",
        default=False,
        action="store_true",
        help="If specified, use the multi-buffer TAG transfer API.",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=12345,
        help="The port the listener will bind to.",
        type=int,
    )

    args = parser.parse_args()
    if args.asyncio_wait and args.progress_mode != "threaded":
        raise RuntimeError("`--asyncio-wait` requires `--progress-mode='threaded'`")

    return args


def main():
    args = parse_args()

    valid_progress_modes = ["blocking", "threaded"]
    if not any(args.progress_mode == v for v in valid_progress_modes):
        raise ValueError(
            f"Unknown progress mode '{args.progress_mode}', "
            f"valid modes are {valid_progress_modes}",
        )

    if args.object_type == "rmm":
        import cupy as xp
        import rmm

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=False,
        )
        xp.cuda.runtime.setDevice(0)
        xp.cuda.set_allocator(rmm.rmm_cupy_allocator)
    else:
        import numpy as xp

    ctx = ucx_api.UCXContext()

    worker = ucx_api.UCXWorker(ctx)

    if args.progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    wireup_send_buf = np.arange(3)
    wireup_recv_buf = np.empty_like(wireup_send_buf)
    send_bufs = [
        xp.arange(50, dtype="u1"),
        xp.arange(500, dtype="u1"),
        xp.arange(50000, dtype="u1"),
    ]

    if args.multi_buffer_transfer is False:
        recv_bufs = [np.empty_like(b) for b in send_bufs]

    global listener_ep
    listener_ep = None

    def listener_callback(conn_request):
        global listener_ep
        listener_ep = listener.create_endpoint_from_conn_request(conn_request, True)

    listener = ucx_api.UCXListener.create(worker, args.port, listener_callback,)

    ep = ucx_api.UCXEndpoint.create(
        worker, "127.0.0.1", args.port, endpoint_error_handling=True,
    )

    while listener_ep is None:
        if args.progress_mode == "blocking":
            worker.progress_worker_event()

    wireup_requests = [
        ep.tag_send(Array(wireup_send_buf), tag=0),
        listener_ep.tag_recv(Array(wireup_recv_buf), tag=0),
    ]
    _wait_requests(worker, args.progress_mode, wireup_requests)

    np.testing.assert_equal(wireup_recv_buf, wireup_send_buf)

    if args.multi_buffer_transfer:
        frames = (
            Array(send_bufs[0]),
            Array(send_bufs[1]),
            Array(send_bufs[2]),
        )
        sizes = tuple(f.nbytes for f in frames)
        is_cuda = tuple(f.cuda for f in frames)

        listener_ep.tag_send_multi_b(frames, sizes, is_cuda, tag=0)
        recv_bufs = ep.tag_recv_multi_b(0)
    else:
        requests = [
            listener_ep.tag_send(Array(send_bufs[0]), tag=0),
            listener_ep.tag_send(Array(send_bufs[1]), tag=1),
            listener_ep.tag_send(Array(send_bufs[2]), tag=2),
            ep.tag_recv(Array(recv_bufs[0]), tag=0),
            ep.tag_recv(Array(recv_bufs[1]), tag=1),
            ep.tag_recv(Array(recv_bufs[2]), tag=2),
        ]

        if args.asyncio_wait:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_wait_requests_async(worker, requests))
        else:
            _wait_requests(worker, args.progress_mode, requests)

            # Check results, raises an exception if any of them failed
            for r in requests:
                r.check_error()

    if args.progress_mode == "threaded":
        worker.stop_progress_thread()

    for recv_buf, send_buf in zip(recv_bufs, send_bufs):
        if args.object_type == "numpy":
            xp.testing.assert_equal(recv_buf, send_buf)
        else:
            xp.testing.assert_array_equal(xp.asarray(recv_buf), send_buf)


if __name__ == "__main__":
    main()
