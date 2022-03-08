# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

def wait_requests(worker, progress_mode, requests):
    if not isinstance(requests, list):
        requests = [requests]

    while not all([r.is_completed() for r in requests]):
        if progress_mode == "blocking":
            worker.progress_worker_event()

    for r in requests:
        r.check_error()
