import multiprocessing as mp
import os

import pytest

from ucxx._lib.arr import Array
import ucxx._lib.libucxx as ucx_api

mp = mp.get_context("spawn")


WireupMessageSize = 10


def _close_callback(closed):
    closed[0] = True


def _wait_requests(worker, progress_mode, requests):
    if not isinstance(requests, list):
        requests = [requests]

    while not all([r.is_completed() for r in requests]):
        if progress_mode == "blocking":
            worker.progress_worker_event()

    for r in requests:
        r.check_error()


def _server(queue, server_close_callback):
    """Server that send received message back to the client

    Notice, since it is illegal to call progress() in call-back functions,
    we use a "chain" of call-back functions.
    """
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    worker = ucx_api.UCXWorker(ctx)

    listener_finished = [False]
    closed = [False]

    # A reference to listener's endpoint is stored to prevent it from going
    # out of scope too early.
    ep = [None]

    def _listener_handler(conn_request):
        ep[0] = listener.create_endpoint_from_conn_request(conn_request, True)
        if server_close_callback is True:
            ep[0].set_close_callback(_close_callback, cb_args=(closed,))
        listener_finished[0] = True

    listener = ucx_api.UCXListener.create(
        worker=worker, port=0, cb_func=_listener_handler
    )
    queue.put(listener.port)

    while ep[0] is None:
        worker.progress()

    wireup_msg = Array(bytearray(WireupMessageSize))
    wireup_request = ep[0].tag_recv(wireup_msg, tag=0)
    _wait_requests(worker, "blocking", wireup_request)

    if server_close_callback is True:
        while closed[0] is False:
            worker.progress()
        assert closed[0] is True
    else:
        while listener_finished[0] is False:
            worker.progress()


def _client(port, server_close_callback):
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create(
        worker, "127.0.0.1", port, endpoint_error_handling=True,
    )
    worker.progress()
    wireup_msg = Array(bytes(os.urandom(WireupMessageSize)))
    wireup_request = ep.tag_send(wireup_msg, tag=0)
    _wait_requests(worker, "blocking", wireup_request)
    if server_close_callback is True:
        ep.close()
        worker.progress()
    else:
        closed = [False]
        ep.set_close_callback(_close_callback, cb_args=(closed,))
        while closed[0] is False:
            worker.progress()


@pytest.mark.parametrize("server_close_callback", [True, False])
def test_close_callback(server_close_callback):
    queue = mp.Queue()
    server = mp.Process(target=_server, args=(queue, server_close_callback),)
    server.start()
    port = queue.get()
    client = mp.Process(target=_client, args=(port, server_close_callback),)
    client.start()
    client.join(timeout=120)
    server.join(timeout=120)
    assert client.exitcode == 0
    assert server.exitcode == 0
