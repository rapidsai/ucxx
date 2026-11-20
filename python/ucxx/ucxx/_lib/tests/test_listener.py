# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import ucxx._lib.libucxx as ucx_api


def test_listener_ip_port():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)

    def _listener_handler(conn_request):
        pass

    listener = ucx_api.UCXListener.create(
        worker=worker, port=0, cb_func=_listener_handler
    )

    assert isinstance(listener.ip, str) and listener.ip
    assert (
        isinstance(listener.port, int) and listener.port >= 0 and listener.port <= 65535
    )


def test_listener_close():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)

    def _listener_handler(conn_request):
        pass

    listener = ucx_api.UCXListener.create(
        worker=worker, port=0, cb_func=_listener_handler
    )
    port = listener.port

    assert not listener.closed
    listener.close()
    assert listener.closed
    assert listener.port == port

    # Closing is idempotent
    listener.close()
    assert listener.closed
