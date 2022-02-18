import argparse
import asyncio
import numpy as np

from ucxx._lib.arr import Array
import ucxx._lib.libucxx as ucx_api


def parse_args():
    parser = argparse.ArgumentParser(description="Basic UCXX-Py Example")
    parser.add_argument(
        "-a",
        "--asyncio",
        default=False,
        action="store_true",
        help="Wait for transfer requests with Python's asyncio.",
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
        "-p",
        "--port",
        default=12345,
        help="The port the listener will bind to.",
        type=int,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    valid_progress_modes = ["blocking", "threaded"]
    if not any(args.progress_mode == v for v in valid_progress_modes):
        raise ValueError(
            f"Unknown progress mode '{args.progress_mode}', "
            f"valid modes are {valid_progress_modes}",
        )

    ctx = ucx_api.UCXContext()

    worker = ucx_api.UCXWorker(ctx)

    if args.progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.startProgressThread()

    wireup_send_buf = np.arange(3)
    wireup_recv_buf = np.empty_like(wireup_send_buf)
    send_bufs = [
        np.arange(50),
        np.arange(500),
        np.arange(50000),
    ]
    recv_bufs = [np.empty_like(b) for b in send_bufs]

    global listener_ep, callback_finished
    listener_ep = None
    callback_finished = False

    def listener_callback(conn_request):
        global listener_ep, callback_finished
        listener_ep = listener.createEndpointFromConnRequest(conn_request, True)
        callback_finished = True

    listener = ucx_api.UCXListener.create(worker, args.port, listener_callback,)

    ep = ucx_api.UCXEndpoint.create(
        worker, "127.0.0.1", args.port, endpoint_error_handling=True,
    )

    while listener_ep is None:
        if args.progress_mode == "blocking":
            worker.progress_worker_event()

    wireup_recv_req = ep.tag_send(Array(wireup_send_buf), tag=0)
    wireup_send_req = listener_ep.tag_recv(Array(wireup_recv_buf), tag=0)

    while not wireup_recv_req.is_completed() or not wireup_send_req.is_completed():
        if args.progress_mode == "blocking":
            worker.progress_worker_event()

    np.testing.assert_equal(wireup_recv_buf, wireup_send_buf)

    requests = [
        listener_ep.tag_send(Array(send_bufs[0]), tag=0),
        listener_ep.tag_send(Array(send_bufs[1]), tag=1),
        listener_ep.tag_send(Array(send_bufs[2]), tag=2),
        ep.tag_recv(Array(recv_bufs[0]), tag=0),
        ep.tag_recv(Array(recv_bufs[1]), tag=1),
        ep.tag_recv(Array(recv_bufs[2]), tag=2),
    ]

    if args.asyncio:
        async def wait_for_requests(requests):
            await asyncio.gather(*[r.is_completed_async() for r in requests])
            # Check results, raises an exception if any of them failed
            for r in requests:
                r.wait()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(wait_for_requests(requests))
    else:
        while not all(r.is_completed() for r in requests):
            if args.progress_mode == "blocking":
                worker.progress_worker_event()
        # Check results, raises an exception if any of them failed
        for r in requests:
            r.wait()

    while callback_finished is not True:
        if args.progress_mode == "blocking":
            worker.progress_worker_event()

    if args.progress_mode == "threaded":
        worker.stopProgressThread()

    for recv_buf, send_buf in zip(recv_bufs, send_bufs):
        np.testing.assert_equal(recv_buf, send_buf)


if __name__ == "__main__":
    main()
