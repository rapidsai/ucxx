from distributed import Scheduler, Worker
from distributed.utils import log_errors

import ucxx

_scheduler_close = Scheduler.close
_worker_close = Worker.close


def _stop_notifier_thread_and_progress_tasks():
    ucxx.stop_notifier_thread()
    ucxx.core._get_ctx().progress_tasks.clear()


async def _scheduler_close_ucxx(*args, **kwargs):
    scheduler = args[0]  # args[0] == self

    await _scheduler_close(*args, **kwargs)

    is_ucxx = any([addr.startswith("ucxx") for addr in scheduler._start_address])

    if is_ucxx:
        _stop_notifier_thread_and_progress_tasks()


@log_errors
async def _worker_close_ucxx(*args, **kwargs):
    # This patch is insufficient for `dask worker` when `--nworkers=1` (default) or
    # `--no-nanny` is specified because there's no good way to detect that the
    # `distributed.Worker.close()` method should stop the notifier thread.

    worker = args[0]  # args[0] == self

    await _worker_close(*args, **kwargs)

    if worker._protocol.startswith("ucxx") and worker.nanny is not None:
        _stop_notifier_thread_and_progress_tasks()


Scheduler.close = _scheduler_close_ucxx
Worker.close = _worker_close_ucxx
