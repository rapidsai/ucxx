import multiprocessing as mp
import os
from queue import Empty as QueueIsEmpty

import pytest

from ucxx._lib.arr import Array
import ucxx._lib.libucxx as ucx_api

mp = mp.get_context("spawn")


def _wait_requests(worker, progress_mode, requests):
    while not all([r.is_completed() for r in requests]):
        if progress_mode == "blocking":
            worker.progress_worker_event()


def _echo_server(get_queue, put_queue, msg_size, progress_mode):
    """Server that send received message back to the client

    Notice, since it is illegal to call progress() in call-back functions,
    we keep a reference to the listener's endpoint and execute tranfers
    outside of the callback function.
    """
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    # A reference to listener's endpoint is stored to prevent it from going
    # out of scope too early.
    global ep, msg, requests
    ep = None
    msg = Array(bytearray(msg_size))
    requests = []

    def _listener_handler(conn_request):
        global ep, msg, requests
        ep = listener.create_endpoint_from_conn_request(conn_request, True)

    listener = ucx_api.UCXListener.create(worker=worker, port=0, cb_func=_listener_handler)
    put_queue.put(listener.port)

    while ep is None:
        if progress_mode == "blocking":
            worker.progress()

    msg = Array(bytearray(msg_size))
    requests = [
        ep.tag_recv(msg, tag=0),
        ep.tag_send(msg, tag=0),
    ]
    _wait_requests(worker, progress_mode, requests)
    for r in requests:
        r.wait()

    while True:
        try:
            get_queue.get(block=True, timeout=0.1)
        except QueueIsEmpty:
            continue
        else:
            break


def _echo_client(msg_size, progress_mode, port):
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    ep = ucx_api.UCXEndpoint.create(
        worker, "127.0.0.1", port, endpoint_error_handling=True,
    )

    if progress_mode == "blocking":
        worker.progress()

    send_msg = bytes(os.urandom(msg_size))
    recv_msg = bytearray(msg_size)
    requests = [
        ep.tag_send(Array(send_msg), tag=0),
        ep.tag_recv(Array(recv_msg), tag=0),
    ]

    _wait_requests(worker, progress_mode, requests)
    assert send_msg == recv_msg


@pytest.mark.parametrize("msg_size", [10, 2 ** 24])
@pytest.mark.parametrize("progress_mode", ["blocking", "threaded"])
def test_server_client(msg_size, progress_mode):
    put_queue, get_queue = mp.Queue(), mp.Queue()
    server = mp.Process(target=_echo_server, args=(put_queue, get_queue, msg_size, progress_mode),)
    server.start()
    port = get_queue.get()
    client = mp.Process(target=_echo_client, args=(msg_size, progress_mode, port))
    client.start()
    client.join(timeout=60)
    assert client.exitcode == 0
    put_queue.put("Finished")
    server.join(timeout=10)
    assert server.exitcode == 0
