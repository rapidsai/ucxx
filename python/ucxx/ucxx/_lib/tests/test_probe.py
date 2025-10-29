# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp

import pytest

from ucxx._lib import libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx.testing import join_processes, terminate_process, wait_requests

mp = mp.get_context("spawn")

WireupMessage = bytearray(b"wireup")
DataMessage = bytearray(b"0" * 10)


def _server_probe(queue, probe_type, api_type="worker"):
    """Server that probes and receives message after client disconnected.

    Note that since it is illegal to call progress() in callback functions,
    we keep a reference to the endpoint after the listener callback has
    terminated, this way we can progress even after Python blocking calls.
    """
    feature_flags = (ucx_api.Feature.AM if probe_type == "am" else ucx_api.Feature.TAG,)
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
    if probe_type == "am":
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
    while ep.alive is True:
        worker.progress()

    # Probe/receive message even after the remote endpoint has disconnected
    if probe_type == "am":
        while ep.am_probe() is False:
            worker.progress()
        recv_req = ep.am_recv()
        wait_requests(worker, "blocking", recv_req)
        received = bytes(recv_req.recv_buffer)
    elif probe_type == "tag":
        if api_type == "worker":
            # Test worker-level API
            while True:
                probe_info = worker.tag_probe(ucx_api.UCXXTag(0))
                if probe_info.matched:
                    break
                worker.progress()
            assert probe_info.sender_tag == ucx_api.UCXXTag(0)
            assert probe_info.length == len(DataMessage)
            assert probe_info.handle is None
            received = bytearray(len(DataMessage))
            wait_requests(
                worker,
                "blocking",
                worker.tag_recv(
                    Array(received),
                    tag=ucx_api.UCXXTag(0),
                    tag_mask=ucx_api.UCXXTagMaskFull,
                ),
            )
        else:
            # Test endpoint-level API (async-style, but using sync API)
            # Note: UCXEndpoint doesn't have tag_probe, so we use worker
            while True:
                probe_info = worker.tag_probe(ucx_api.UCXXTag(0))
                if probe_info.matched:
                    break
                worker.progress()
            assert probe_info.sender_tag == ucx_api.UCXXTag(0)
            assert probe_info.length == len(DataMessage)
            assert probe_info.handle is None
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
    elif probe_type == "tag_remove":
        if api_type == "worker":
            # Test worker-level API with remove=True
            while True:
                probe_info = worker.tag_probe(ucx_api.UCXXTag(0), remove=True)
                if probe_info.matched:
                    break
                worker.progress()
            assert probe_info.sender_tag == ucx_api.UCXXTag(0)
            assert probe_info.length == len(DataMessage)
            assert probe_info.handle
            received = bytearray(len(DataMessage))
            wait_requests(
                worker,
                "blocking",
                worker.tag_recv_with_handle(Array(received), probe_info),
            )
        else:
            # Test endpoint-level API with remove=True
            # Note: UCXEndpoint doesn't have tag_probe, so we use worker
            while True:
                probe_info = worker.tag_probe(ucx_api.UCXXTag(0), remove=True)
                if probe_info.matched:
                    break
                worker.progress()
            assert probe_info.sender_tag == ucx_api.UCXXTag(0)
            assert probe_info.length == len(DataMessage)
            assert probe_info.handle
            received = bytearray(len(DataMessage))
            wait_requests(
                worker,
                "blocking",
                ep.tag_recv_with_handle(Array(received), probe_info),
            )

    assert wireup == WireupMessage
    assert received == DataMessage


def _client_probe(queue, probe_type):
    feature_flags = (ucx_api.Feature.AM if probe_type == "am" else ucx_api.Feature.TAG,)
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)
    port = queue.get()
    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )

    if probe_type == "am":
        requests = [
            ep.am_send(Array(WireupMessage)),
            ep.am_send(Array(DataMessage)),
        ]
    elif probe_type == "tag":
        requests = [
            ep.tag_send(Array(WireupMessage), tag=ucx_api.UCXXTag(0)),
            ep.tag_send(Array(DataMessage), tag=ucx_api.UCXXTag(0)),
        ]
    elif probe_type == "tag_remove":
        requests = [
            ep.tag_send(Array(WireupMessage), tag=ucx_api.UCXXTag(0)),
            ep.tag_send(Array(DataMessage), tag=ucx_api.UCXXTag(0)),
        ]
    wait_requests(worker, "blocking", requests)

    # Wait for wireup before disconnecting
    assert queue.get() == "wireup completed"


@pytest.mark.parametrize("probe_type", ["am", "tag", "tag_remove"])
@pytest.mark.parametrize("api_type", ["worker", "endpoint"])
def test_message_probe(probe_type, api_type):
    queue = mp.Queue()
    server = mp.Process(target=_server_probe, args=(queue, probe_type, api_type))
    server.start()
    client = mp.Process(target=_client_probe, args=(queue, probe_type))
    client.start()
    join_processes([client, server], timeout=60)
    terminate_process(client)
    terminate_process(server)
