import multiprocessing as mp

import pytest

from ucxx._lib.arr import Array
import ucxx._lib.libucxx as ucx_api

mp = mp.get_context("spawn")


def _server_cancel(queue):
    """Server that establishes an endpoint to client and immediately closes
    it, triggering received messages to be canceled on the client.
    """
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)

    # Keep endpoint to be used from outside the listener callback
    ep = [None]

    def _listener_handler(conn_request):
        ep[0] = listener.create_endpoint_from_conn_request(conn_request, True)

    listener = ucx_api.UCXListener.create(worker=worker, port=0, cb_func=_listener_handler)
    queue.put(listener.port)

    while ep[0] is None:
        worker.progress()


def _client_cancel(queue):
    """Client that connects to server and waits for messages to be received,
    because the server closes without sending anything, the messages will
    trigger cancelation.
    """
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    port = queue.get()
    ep = ucx_api.UCXEndpoint.create(
        worker, "127.0.0.1", port, endpoint_error_handling=True,
    )

    msg = Array(bytearray(1))
    request = ep.tag_recv(msg, tag=0)

    canceled = worker.cancel_inflight_requests()
    while canceled == 0:
        worker.progress()
        canceled = worker.cancel_inflight_requests()

    with pytest.raises(ucx_api.UCXCanceled):
        request.check_error()

    assert canceled == 1


def test_message_probe():
    queue = mp.Queue()
    server = mp.Process(target=_server_cancel, args=(queue, ),)
    server.start()
    client = mp.Process(target=_client_cancel, args=(queue, ),)
    client.start()
    client.join(timeout=10)
    server.join(timeout=10)
    assert client.exitcode == 0
    assert server.exitcode == 0
