import multiprocessing as mp
import os
from queue import Empty as QueueIsEmpty

import pytest

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx.testing import wait_requests

mp = mp.get_context("spawn")


WireupMessageSize = 10


def _send(ep, api, message):
    if api == "stream":
        return ep.stream_send(message)
    else:
        return ep.tag_send(message, tag=0)


def _recv(ep, api, message):
    if api == "stream":
        return ep.stream_recv(message)
    else:
        return ep.tag_recv(message, tag=0)


def _echo_server(get_queue, put_queue, transfer_api, msg_size, progress_mode):
    """Server that send received message back to the client

    Notice, since it is illegal to call progress() in call-back functions,
    we keep a reference to the listener's endpoint and execute tranfers
    outside of the callback function.
    """
    feature_flags = [ucx_api.Feature.WAKEUP]
    if transfer_api == "stream":
        feature_flags.append(ucx_api.Feature.STREAM)
    else:
        feature_flags.append(ucx_api.Feature.TAG)

    ctx = ucx_api.UCXContext(feature_flags=tuple(feature_flags))
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    # A reference to listener's endpoint is stored to prevent it from going
    # out of scope too early and allow transfers outside of the listsner's
    # callback even after it has terminated.
    ep = [None]

    def _listener_handler(conn_request):
        ep[0] = listener.create_endpoint_from_conn_request(conn_request, True)

    listener = ucx_api.UCXListener.create(
        worker=worker, port=0, cb_func=_listener_handler
    )
    put_queue.put(listener.port)

    while ep[0] is None:
        if progress_mode == "blocking":
            worker.progress()

    wireup_msg = Array(bytearray(WireupMessageSize))
    wireup_request = _recv(ep[0], transfer_api, wireup_msg)
    wait_requests(worker, progress_mode, wireup_request)

    msg = Array(bytearray(msg_size))

    # We reuse the message buffer, so we must receive, wait, and then send
    # it back again.
    requests = [_recv(ep[0], transfer_api, msg)]
    wait_requests(worker, progress_mode, requests)
    requests = [_send(ep[0], transfer_api, msg)]
    wait_requests(worker, progress_mode, requests)

    while True:
        try:
            get_queue.get(block=True, timeout=0.1)
        except QueueIsEmpty:
            continue
        else:
            break


def _echo_client(transfer_api, msg_size, progress_mode, port):
    feature_flags = [ucx_api.Feature.WAKEUP]
    if transfer_api == "stream":
        feature_flags.append(ucx_api.Feature.STREAM)
    else:
        feature_flags.append(ucx_api.Feature.TAG)

    ctx = ucx_api.UCXContext(feature_flags=tuple(feature_flags))
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

    wireup_msg = Array(bytes(os.urandom(WireupMessageSize)))
    wireup_request = _send(ep, transfer_api, wireup_msg)
    wait_requests(worker, progress_mode, wireup_request)

    send_msg = bytes(os.urandom(msg_size))
    recv_msg = bytearray(msg_size)
    requests = [
        _send(ep, transfer_api, Array(send_msg)),
        _recv(ep, transfer_api, Array(recv_msg)),
    ]
    wait_requests(worker, progress_mode, requests)

    assert recv_msg == send_msg


@pytest.mark.parametrize("transfer_api", ["stream", "tag"])
@pytest.mark.parametrize("msg_size", [10, 2 ** 24])
@pytest.mark.parametrize("progress_mode", ["blocking", "threaded"])
def test_server_client(transfer_api, msg_size, progress_mode):
    put_queue, get_queue = mp.Queue(), mp.Queue()
    server = mp.Process(
        target=_echo_server,
        args=(put_queue, get_queue, transfer_api, msg_size, progress_mode),
    )
    server.start()
    port = get_queue.get()
    client = mp.Process(
        target=_echo_client, args=(transfer_api, msg_size, progress_mode, port)
    )
    client.start()
    client.join(timeout=60)
    assert client.exitcode == 0
    put_queue.put("Finished")
    server.join(timeout=10)
    assert server.exitcode == 0
