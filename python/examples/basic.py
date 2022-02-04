import numpy as np

from ucxx._lib.arr import Array
import ucxx._lib.libucxx as ucx_api


def main():
    ctx = ucx_api.UCXContext()

    worker = ucx_api.UCXWorker(ctx)

    worker.startProgressThread()

    send_buf = np.arange(10)
    recv_buf = np.empty_like(send_buf)

    global listener_ep, callback_finished
    listener_ep = None
    callback_finished = False

    def listener_callback(conn_request):
        global listener_ep, callback_finished
        listener_ep = listener.createEndpointFromConnRequest(conn_request, True)
        callback_finished = True

    listener = ucx_api.UCXListener.create(worker, 12345, listener_callback,)

    ep = ucx_api.UCXEndpoint.create(
        worker, "127.0.0.1", 12345, endpoint_error_handling=True,
    )

    while listener_ep is None:
        pass

    ep.tag_send(Array(send_buf), tag=0)
    listener_ep.tag_recv(Array(recv_buf), tag=0)

    while callback_finished is not True:
        pass

    worker.stopProgressThread()

    np.testing.assert_equal(recv_buf, send_buf)


main()
