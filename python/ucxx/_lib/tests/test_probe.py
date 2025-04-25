# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp

from ucxx._lib import libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx.testing import terminate_process, wait_requests

mp = mp.get_context("spawn")

WireupMessage = bytearray(b"wireup")
DataMessage = bytearray(b"0" * 10)


def _server_probe(queue):
    """Server that probes and receives message after client disconnected.

    Note that since it is illegal to call progress() in callback functions,
    we keep a reference to the endpoint after the listener callback has
    terminated, this way we can progress even after Python blocking calls.
    """
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    worker = ucx_api.UCXWorker(ctx)

    # Keep endpoint to be used from outside the listener callback
    ep = [None]

    def _listener_handler(conn_request):
        ep[0] = listener.create_endpoint_from_conn_request(
            conn_request, endpoint_error_handling=True
        )

    listener = ucx_api.UCXListener.create(
        worker=worker, port=0, endpoint_error_handling=True, cb_func=_listener_handler
    )
    queue.put(listener.port)

    while ep[0] is None:
        worker.progress()

    ep = ep[0]

    # Ensure wireup and inform client before it can disconnect
    wireup = bytearray(len(WireupMessage))
    wait_requests(worker, "blocking", ep.tag_recv(Array(wireup), tag=0))
    queue.put("wireup completed")

    # Ensure client has disconnected -- endpoint is not alive anymore
    while ep.is_alive() is True:
        worker.progress()

    # Probe/receive message even after the remote endpoint has disconnected
    while worker.tag_probe(0) is False:
        worker.progress()
    received = bytearray(len(DataMessage))
    wait_requests(worker, "blocking", ep.tag_recv(Array(received), tag=0))

    assert wireup == WireupMessage
    assert received == DataMessage


def _client_probe(queue):
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    worker = ucx_api.UCXWorker(ctx)
    port = queue.get()
    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )

    requests = [
        ep.tag_send(Array(WireupMessage), tag=0),
        ep.tag_send(Array(DataMessage), tag=0),
    ]
    wait_requests(worker, "blocking", requests)

    # Wait for wireup before disconnecting
    assert queue.get() == "wireup completed"


def test_message_probe():
    queue = mp.Queue()
    server = mp.Process(target=_server_probe, args=(queue,))
    server.start()
    client = mp.Process(target=_client_probe, args=(queue,))
    client.start()
    client.join(timeout=10)
    server.join(timeout=10)
    terminate_process(client)
    terminate_process(server)
