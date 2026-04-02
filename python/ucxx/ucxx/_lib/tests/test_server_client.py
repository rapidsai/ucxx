# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp
import os
from queue import Empty as QueueIsEmpty

import pytest

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx.testing import terminate_process, wait_requests

mp = mp.get_context("spawn")


WireupMessageSize = 10


def _send(ep, api, message, memory_type_policy=None):
    if api == "am":
        return ep.am_send(message, memory_type_policy=memory_type_policy)
    elif api == "stream":
        return ep.stream_send(message)
    else:
        return ep.tag_send(message, tag=ucx_api.UCXXTag(0))


def _recv(ep, api, message):
    if api == "am":
        return ep.am_recv()
    elif api == "stream":
        return ep.stream_recv(message)
    else:
        return ep.tag_recv(message, tag=ucx_api.UCXXTag(0))


def _echo_server(get_queue, put_queue, transfer_api, msg_size, progress_mode):
    """Server that send received message back to the client

    Notice, since it is illegal to call progress() in call-back functions,
    we keep a reference to the listener's endpoint and execute transfers
    outside of the callback function.
    """
    # TAG is always used for wireup
    feature_flags = [ucx_api.Feature.WAKEUP]
    if transfer_api == "am":
        feature_flags.append(ucx_api.Feature.AM)
    elif transfer_api == "stream":
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

    wireup_msg_recv = Array(bytearray(WireupMessageSize))
    wireup_msg_send = Array(bytes(os.urandom(WireupMessageSize)))
    wireup_requests = [
        _recv(ep[0], transfer_api, wireup_msg_recv),
        _send(ep[0], transfer_api, wireup_msg_send),
    ]
    wait_requests(worker, progress_mode, wireup_requests)

    msg = Array(bytearray(msg_size))

    if transfer_api == "stream" and msg_size == 0:
        with pytest.raises(RuntimeError):
            _recv(ep[0], transfer_api, msg)
        with pytest.raises(RuntimeError):
            _send(ep[0], transfer_api, msg)
        return

    # We reuse the message buffer, so we must receive, wait, and then send
    # it back again.
    requests = [_recv(ep[0], transfer_api, msg)]
    wait_requests(worker, progress_mode, requests)
    if transfer_api == "am":
        msg = Array(requests[0].recv_buffer)
    requests = [_send(ep[0], transfer_api, msg)]
    wait_requests(worker, progress_mode, requests)

    while True:
        try:
            get_queue.get(block=True, timeout=0.1)
        except QueueIsEmpty:
            continue
        else:
            break

    if progress_mode == "thread":
        worker.stop_progress_thread()


def _echo_client(transfer_api, msg_size, progress_mode, port):
    # TAG is always used for wireup
    feature_flags = [ucx_api.Feature.WAKEUP]
    if transfer_api == "am":
        feature_flags.append(ucx_api.Feature.AM)
    elif transfer_api == "stream":
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
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )

    if progress_mode == "blocking":
        worker.progress()

    wireup_msg_send = Array(bytes(os.urandom(WireupMessageSize)))
    wireup_msg_recv = Array(bytearray(WireupMessageSize))
    wireup_requests = [
        _send(ep, transfer_api, wireup_msg_send),
        _recv(ep, transfer_api, wireup_msg_recv),
    ]
    wait_requests(worker, progress_mode, wireup_requests)

    send_msg = bytes(os.urandom(msg_size))
    recv_msg = bytearray(msg_size)

    if transfer_api == "stream" and msg_size == 0:
        with pytest.raises(RuntimeError):
            _send(ep, transfer_api, Array(send_msg))
        with pytest.raises(RuntimeError):
            _recv(ep, transfer_api, Array(recv_msg))
        return

    requests = [
        _send(ep, transfer_api, Array(send_msg)),
        _recv(ep, transfer_api, Array(recv_msg)),
    ]
    wait_requests(worker, progress_mode, requests)

    if transfer_api == "am":
        recv_msg = requests[1].recv_buffer

        assert bytes(recv_msg) == send_msg
    else:
        assert recv_msg == send_msg

    if progress_mode == "thread":
        worker.stop_progress_thread()


def _echo_server_am_params(
    get_queue, put_queue, msg_size, progress_mode, memory_type_policy
):
    """Server that echoes AM messages using the AmSendParams code path."""
    feature_flags = (ucx_api.Feature.WAKEUP, ucx_api.Feature.AM)
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

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

    msg = Array(bytearray(msg_size))
    requests = [ep[0].am_recv()]
    wait_requests(worker, progress_mode, requests)
    msg = Array(requests[0].recv_buffer)
    requests = [ep[0].am_send(msg, memory_type_policy=memory_type_policy)]
    wait_requests(worker, progress_mode, requests)

    while True:
        try:
            get_queue.get(block=True, timeout=0.1)
        except QueueIsEmpty:
            continue
        else:
            break

    if progress_mode == "thread":
        worker.stop_progress_thread()


def _echo_client_am_params(msg_size, progress_mode, memory_type_policy, port):
    """Client that sends and receives AM messages using AmSendParams."""
    feature_flags = (ucx_api.Feature.WAKEUP, ucx_api.Feature.AM)
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )

    if progress_mode == "blocking":
        worker.progress()

    send_msg = bytes(os.urandom(msg_size))

    requests = [
        ep.am_send(Array(send_msg), memory_type_policy=memory_type_policy),
        ep.am_recv(),
    ]
    wait_requests(worker, progress_mode, requests)

    recv_msg = requests[1].recv_buffer
    assert bytes(recv_msg) == send_msg

    if progress_mode == "thread":
        worker.stop_progress_thread()


@pytest.mark.parametrize(
    "memory_type_policy",
    [None, ucx_api.PythonAmSendMemoryTypePolicy.FallbackToHost],
)
@pytest.mark.parametrize("msg_size", [10, 2**24])
@pytest.mark.parametrize("progress_mode", ["blocking", "thread"])
def test_server_client_am_params(msg_size, progress_mode, memory_type_policy):
    put_queue, get_queue = mp.Queue(), mp.Queue()
    server = mp.Process(
        target=_echo_server_am_params,
        args=(put_queue, get_queue, msg_size, progress_mode, memory_type_policy),
    )
    server.start()
    port = get_queue.get()
    client = mp.Process(
        target=_echo_client_am_params,
        args=(msg_size, progress_mode, memory_type_policy, port),
    )
    client.start()
    client.join(timeout=60)
    terminate_process(client)
    put_queue.put("Finished")
    server.join(timeout=10)
    terminate_process(server)


def _echo_server_am_iov(get_queue, put_queue, msg_size, progress_mode):
    """Server that receives an IOV AM message and echoes it back."""
    feature_flags = (ucx_api.Feature.WAKEUP, ucx_api.Feature.AM)
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

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

    # Receive the IOV message (arrives as a single contiguous buffer)
    requests = [ep[0].am_recv()]
    wait_requests(worker, progress_mode, requests)
    msg = Array(requests[0].recv_buffer)
    # Echo back as a regular contiguous send
    requests = [ep[0].am_send(msg)]
    wait_requests(worker, progress_mode, requests)

    while True:
        try:
            get_queue.get(block=True, timeout=0.1)
        except QueueIsEmpty:
            continue
        else:
            break

    if progress_mode == "thread":
        worker.stop_progress_thread()


def _echo_client_am_iov(msg_size, progress_mode, port):
    """Client that sends an IOV AM message and receives the echo."""
    feature_flags = (ucx_api.Feature.WAKEUP, ucx_api.Feature.AM)
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )

    if progress_mode == "blocking":
        worker.progress()

    send_msg = bytes(os.urandom(msg_size))

    # Split the message into two segments for IOV send
    mid = msg_size // 2
    seg1 = Array(send_msg[:mid])
    seg2 = Array(send_msg[mid:])

    requests = [
        ep.am_send_iov([seg1, seg2]),
        ep.am_recv(),
    ]
    wait_requests(worker, progress_mode, requests)

    recv_msg = requests[1].recv_buffer
    assert bytes(recv_msg) == send_msg

    if progress_mode == "thread":
        worker.stop_progress_thread()


@pytest.mark.parametrize("msg_size", [10, 2**24])
@pytest.mark.parametrize("progress_mode", ["blocking", "thread"])
def test_server_client_am_iov(msg_size, progress_mode):
    put_queue, get_queue = mp.Queue(), mp.Queue()
    server = mp.Process(
        target=_echo_server_am_iov,
        args=(put_queue, get_queue, msg_size, progress_mode),
    )
    server.start()
    port = get_queue.get()
    client = mp.Process(
        target=_echo_client_am_iov,
        args=(msg_size, progress_mode, port),
    )
    client.start()
    client.join(timeout=60)
    terminate_process(client)
    put_queue.put("Finished")
    server.join(timeout=10)
    terminate_process(server)


def _user_header_server(get_queue, put_queue, msg_size, progress_mode):
    """Server that receives an AM message with user header and echoes both back."""
    feature_flags = (ucx_api.Feature.WAKEUP, ucx_api.Feature.AM)
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

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

    # Receive the message and its user header
    requests = [ep[0].am_recv()]
    wait_requests(worker, progress_mode, requests)
    recv_header = requests[0].recv_header
    msg = Array(requests[0].recv_buffer)

    # Echo back with the same user header
    requests = [ep[0].am_send(msg, user_header=recv_header)]
    wait_requests(worker, progress_mode, requests)

    while True:
        try:
            get_queue.get(block=True, timeout=0.1)
        except QueueIsEmpty:
            continue
        else:
            break

    if progress_mode == "thread":
        worker.stop_progress_thread()


def _user_header_client(msg_size, progress_mode, port):
    """Client that sends AM with user header and validates the echo."""
    feature_flags = (ucx_api.Feature.WAKEUP, ucx_api.Feature.AM)
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )

    if progress_mode == "blocking":
        worker.progress()

    send_msg = bytes(os.urandom(msg_size))
    user_header = b"test-header-\x00\x01\xff"

    requests = [
        ep.am_send(Array(send_msg), user_header=user_header),
        ep.am_recv(),
    ]
    wait_requests(worker, progress_mode, requests)

    recv_msg = requests[1].recv_buffer
    recv_header = requests[1].recv_header
    assert bytes(recv_msg) == send_msg
    assert recv_header == user_header

    if progress_mode == "thread":
        worker.stop_progress_thread()


def _user_header_iov_client(msg_size, progress_mode, port):
    """Client that sends IOV AM with user header and validates the echo."""
    feature_flags = (ucx_api.Feature.WAKEUP, ucx_api.Feature.AM)
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )

    if progress_mode == "blocking":
        worker.progress()

    send_msg = bytes(os.urandom(msg_size))
    user_header = b"iov-header-data"

    mid = msg_size // 2
    seg1 = Array(send_msg[:mid])
    seg2 = Array(send_msg[mid:])

    requests = [
        ep.am_send_iov([seg1, seg2], user_header=user_header),
        ep.am_recv(),
    ]
    wait_requests(worker, progress_mode, requests)

    recv_msg = requests[1].recv_buffer
    recv_header = requests[1].recv_header
    assert bytes(recv_msg) == send_msg
    assert recv_header == user_header

    if progress_mode == "thread":
        worker.stop_progress_thread()


def _empty_user_header_client(msg_size, progress_mode, port):
    """Client that sends AM without user header and validates empty recv_header."""
    feature_flags = (ucx_api.Feature.WAKEUP, ucx_api.Feature.AM)
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )

    if progress_mode == "blocking":
        worker.progress()

    send_msg = bytes(os.urandom(msg_size))

    requests = [
        ep.am_send(Array(send_msg)),
        ep.am_recv(),
    ]
    wait_requests(worker, progress_mode, requests)

    recv_msg = requests[1].recv_buffer
    recv_header = requests[1].recv_header
    assert bytes(recv_msg) == send_msg
    assert recv_header == b""

    if progress_mode == "thread":
        worker.stop_progress_thread()


@pytest.mark.parametrize("msg_size", [10, 2**24])
@pytest.mark.parametrize("progress_mode", ["blocking", "thread"])
def test_server_client_am_user_header(msg_size, progress_mode):
    put_queue, get_queue = mp.Queue(), mp.Queue()
    server = mp.Process(
        target=_user_header_server,
        args=(put_queue, get_queue, msg_size, progress_mode),
    )
    server.start()
    port = get_queue.get()
    client = mp.Process(
        target=_user_header_client,
        args=(msg_size, progress_mode, port),
    )
    client.start()
    client.join(timeout=60)
    terminate_process(client)
    put_queue.put("Finished")
    server.join(timeout=10)
    terminate_process(server)


@pytest.mark.parametrize("msg_size", [10, 2**24])
@pytest.mark.parametrize("progress_mode", ["blocking", "thread"])
def test_server_client_am_iov_user_header(msg_size, progress_mode):
    put_queue, get_queue = mp.Queue(), mp.Queue()
    server = mp.Process(
        target=_user_header_server,
        args=(put_queue, get_queue, msg_size, progress_mode),
    )
    server.start()
    port = get_queue.get()
    client = mp.Process(
        target=_user_header_iov_client,
        args=(msg_size, progress_mode, port),
    )
    client.start()
    client.join(timeout=60)
    terminate_process(client)
    put_queue.put("Finished")
    server.join(timeout=10)
    terminate_process(server)


@pytest.mark.parametrize("msg_size", [10, 2**24])
@pytest.mark.parametrize("progress_mode", ["blocking", "thread"])
def test_server_client_am_empty_user_header(msg_size, progress_mode):
    """Test that recv_header is empty bytes when no user header is sent."""
    put_queue, get_queue = mp.Queue(), mp.Queue()
    # Reuse the echo server that doesn't set user_header
    server = mp.Process(
        target=_echo_server_am_params,
        args=(put_queue, get_queue, msg_size, progress_mode, None),
    )
    server.start()
    port = get_queue.get()
    client = mp.Process(
        target=_empty_user_header_client,
        args=(msg_size, progress_mode, port),
    )
    client.start()
    client.join(timeout=60)
    terminate_process(client)
    put_queue.put("Finished")
    server.join(timeout=10)
    terminate_process(server)


@pytest.mark.parametrize("transfer_api", ["am", "stream", "tag"])
@pytest.mark.parametrize("msg_size", [0, 10, 2**24])
@pytest.mark.parametrize("progress_mode", ["blocking", "thread"])
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
    terminate_process(client)
    put_queue.put("Finished")
    server.join(timeout=10)
    terminate_process(server)
