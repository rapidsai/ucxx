from distributed import Nanny, Scheduler, Worker
from distributed.utils import log_errors

import ucxx

_scheduler_init = Scheduler.__init__
_scheduler_start_unsafe = Scheduler.start_unsafe
_scheduler_close = Scheduler.close
_worker_init = Worker.__init__
_worker_start_unsafe = Worker.start_unsafe
_worker_close = Worker.close
_nanny_init = Nanny.__init__
_nanny_start_unsafe = Nanny.start_unsafe
_nanny_close = Nanny.close


def _stop_notifier_thread_and_progress_tasks():
    """Stop the notifier thread and progress tasks.

    If no Dask resources that make use of UCXX communicator are running anymore,
    stop the notifier thread and progress tasks to allow for clean shutdown.
    """
    ctx = ucxx.core._get_ctx()
    if len(ctx._dask_resources) == 0:
        ucxx.stop_notifier_thread()
        ucxx.core._get_ctx().progress_tasks.clear()


def _register_dask_resource(resource, name=None):
    """Register a Dask resource with the UCXX context.

    Register a Dask resource with the UCXX context to keep track of it, so that
    the notifier thread and progress tasks may be stopped when no more resources
    need UCXX.
    """
    if name is None:
        name = "Unknown caller"

    ctx = ucxx.core._get_ctx()

    with ctx._dask_resources_lock:
        ctx._dask_resources.add(resource)


def _deregister_dask_resource(resource, name=None):
    """Deregister a Dask resource with the UCXX context.

    Deregister a Dask resource from the UCXX context, and if no resources remain
    after deregistration, stop the notifier thread and progress tasks.
    need UCXX.
    """
    if name is None:
        name = "Unknown caller"

    ctx = ucxx.core._get_ctx()

    with ctx._dask_resources_lock:
        try:
            ctx._dask_resources.remove(resource)
        except KeyError:
            pass
        _stop_notifier_thread_and_progress_tasks()


async def _scheduler_start_unsafe_ucxx(self, *args, **kwargs):
    """Start `Scheduler` and register with UCXX.

    Start the `Scheduler` instance and register the resource with the UCXX
    context so that its lifetime may be tracked.
    """
    await _scheduler_start_unsafe(self, *args, **kwargs)
    _register_dask_resource(self, "ucxx.Scheduler.start_unsafe")


async def _scheduler_close_ucxx(self, *args, **kwargs):
    """Close `Scheduler` and deregister from UCXX.

    Close the `Scheduler` instance and deregister the resource from the UCXX
    context. If this is the last object registered, stop the notifier thread
    and progress tasks.
    """
    await _scheduler_close(self, *args, **kwargs)

    is_ucxx = any([addr.startswith("ucxx") for addr in self._start_address])

    if is_ucxx:
        _deregister_dask_resource(self, "ucxx.Scheduler.close")


async def _worker_start_unsafe_ucxx(*args, **kwargs):
    """Start `Worker` and register with UCXX.

    Start the `Worker` instance and register the resource with the UCXX
    context so that its lifetime may be tracked.

    .. note::
        This only applies for the current process, if launching a `Worker` from
        `Nanny`, the new subprocess must patch it again. For that purpose,
        `UcxxWorker` is necessary instead of this.
    """
    await _worker_start_unsafe(*args, **kwargs)


@log_errors
async def _worker_close_ucxx(self, *args, **kwargs):
    """Close `Worker` and deregister from UCXX.

    Close the `Worker` instance and deregister the resource from the UCXX
    context. If this is the last object registered, stop the notifier thread
    and progress tasks.

    .. note::
        This only applies for the current process, if launching a `Worker` from
        `Nanny`, the new subprocess must patch it again. For that purpose,
        `UcxxWorker` is necessary instead of this.
    """
    await _worker_close(self, *args, **kwargs)

    if self._protocol.startswith("ucxx"):
        _deregister_dask_resource(self, "ucxx.Worker.close")


class UcxxWorker(Worker):
    """Subclass of `Worker` to allow tracking lifetime with UCXX.

    This subclass is required when a `Worker` is spawned from `Nanny`. Since
    `Nanny` spawns a new subprocess, monkey-patching of `Worker` does not get
    passed through to the new process, and when the patching occurs, UCXX
    initialization is already done.
    """

    async def start_unsafe(self, *args, **kwargs):
        """Start `Worker` and register with UCXX.

        Start the `Worker` instance and register the resource with the UCXX
        context so that its lifetime may be tracked.
        """
        await super().start_unsafe(*args, **kwargs)
        _register_dask_resource(self, "UcxxWorker.start_unsafe")

    @log_errors
    async def close(self, *args, **kwargs):
        """Close `Worker` and deregister from UCXX.

        Close the `Worker` instance and deregister the resource from the UCXX
        context. If this is the last object registered, stop the notifier thread
        and progress tasks.
        """
        await super().close(*args, **kwargs)

        if self._protocol.startswith("ucxx") and self.nanny is not None:
            _deregister_dask_resource(self, "UcxxWorker.close")


def _nanny_init_ucxx(self, *args, **kwargs):
    """`Nanny` with custom `UcxxWorker`.

    Initialize a `Nanny` with the custom `UcxxWorker` that registers itself with
    the UCXX context so that its lifetime may be tracked.
    """
    if "worker_class" in kwargs and kwargs["worker_class"] is not Worker:
        raise ValueError("`Nanny` with custom 'worker_class' is not supported yet.")
    else:
        worker_class = UcxxWorker
        kwargs["worker_class"] = worker_class

    _nanny_init(self, *args, **kwargs)


async def _nanny_start_unsafe_ucxx(self, *args, **kwargs):
    """Start `Nanny` and register with UCXX.

    Start the `Nanny` instance and register the resource with the UCXX
    context so that its lifetime may be tracked.
    """
    await _nanny_start_unsafe(self, *args, **kwargs)
    _register_dask_resource(self, "ucxx.Nanny.start_unsafe")


async def _nanny_close_ucxx(self, *args, **kwargs):
    """Close `Worker` and deregister from UCXX.

    Close the `Worker` instance and deregister the resource from the UCXX
    context. If this is the last object registered, stop the notifier thread
    and progress tasks.
    """
    await _nanny_close(self, *args, **kwargs)

    if self._protocol.startswith("ucxx"):
        _deregister_dask_resource(self, "ucxx.Nanny.close")


Scheduler.close = _scheduler_close_ucxx
Scheduler.start_unsafe = _scheduler_start_unsafe_ucxx
Nanny.__init__ = _nanny_init_ucxx
Nanny.start_unsafe = _nanny_start_unsafe_ucxx
Nanny.close = _nanny_close_ucxx
