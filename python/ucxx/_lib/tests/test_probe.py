# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp

import pytest
from ucxx._lib import libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx.testing import terminate_process, wait_requests

mp = mp.get_context("spawn")

WireupMessage = bytearray(b"wireup")
DataMessage = bytearray(b"0" * 10)


def _server_probe(queue, transfer_api):
    """Server that probes and receives message after client disconnected.

    Note that since it is illegal to call progress() in callback functions,
    we keep a reference to the endpoint after the listener callback has
    terminated, this way we can progress even after Python blocking calls.
    """
    feature_flags = (
        ucx_api.Feature.AM if transfer_api == "am" else ucx_api.Feature.TAG,
    )
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)

    # Keep endpoint to be used from outside the listener callback
    ep = [None]

    def _listener_handler(conn_request):
        ep[0] = listener.create_endpoint_from_conn_request(
            conn_request, endpoint_error_handling=True
        )

    listener = ucx_api.UCXListener.create(
        worker=worker, port=0, cb_func=_listener_handler
    )
    queue.put(listener.port)

    while ep[0] is None:
        worker.progress()

    ep = ep[0]

    # Ensure wireup and inform client before it can disconnect
    if transfer_api == "am":
        wireup_req = ep.am_recv()
        wait_requests(worker, "blocking", wireup_req)
        wireup = bytes(wireup_req.recv_buffer)
    else:
        wireup = bytearray(len(WireupMessage))
        wait_requests(
            worker,
            "blocking",
            ep.tag_recv(
                Array(wireup), tag=ucx_api.UCXXTag(0), tag_mask=ucx_api.UCXXTagMaskFull
            ),
        )
    queue.put("wireup completed")

    # Ensure client has disconnected -- endpoint is not alive anymore
    while ep.is_alive() is True:
        worker.progress()

    # Probe/receive message even after the remote endpoint has disconnected
    if transfer_api == "am":
        while ep.am_probe() is False:
            worker.progress()
        recv_req = ep.am_recv()
        wait_requests(worker, "blocking", recv_req)
        received = bytes(recv_req.recv_buffer)
    else:
        while worker.tag_probe(ucx_api.UCXXTag(0)) is False:
            worker.progress()
        received = bytearray(len(DataMessage))
        wait_requests(
            worker,
            "blocking",
            ep.tag_recv(
                Array(received),
                tag=ucx_api.UCXXTag(0),
                tag_mask=ucx_api.UCXXTagMaskFull,
            ),
        )

    assert wireup == WireupMessage
    assert received == DataMessage


def _client_probe(queue, transfer_api):
    feature_flags = (
        ucx_api.Feature.AM if transfer_api == "am" else ucx_api.Feature.TAG,
    )
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)
    port = queue.get()
    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )

    if transfer_api == "am":
        requests = [
            ep.am_send(Array(WireupMessage)),
            ep.am_send(Array(DataMessage)),
        ]
    else:
        requests = [
            ep.tag_send(Array(WireupMessage), tag=ucx_api.UCXXTag(0)),
            ep.tag_send(Array(DataMessage), tag=ucx_api.UCXXTag(0)),
        ]
    wait_requests(worker, "blocking", requests)

    # Wait for wireup before disconnecting
    assert queue.get() == "wireup completed"


@pytest.mark.parametrize("transfer_api", ["am", "tag"])
def test_message_probe(transfer_api):
    queue = mp.Queue()
    server = mp.Process(target=_server_probe, args=(queue, transfer_api))
    server.start()
    client = mp.Process(target=_client_probe, args=(queue, transfer_api))
    client.start()
    client.join(timeout=10)
    server.join(timeout=10)
    terminate_process(client)
    terminate_process(server)
