import pytest

from distributed import Client, Nanny
from distributed.scheduler import Scheduler
from distributed.utils_test import gen_test
from distributed.worker import Worker


@pytest.mark.parametrize("Worker", [Worker, Nanny])
@gen_test()
async def test_protocol_from_scheduler_address(ucxx_loop, Worker):
    pytest.importorskip("ucxx")

    async with Scheduler(protocol="ucxx", dashboard_address=":0") as s:
        assert s.address.startswith("ucxx://")
        async with Worker(s.address) as w:
            assert w.address.startswith("ucxx://")
            async with Client(s.address, asynchronous=True) as c:
                info = c.scheduler_info()
                assert info["address"].startswith("ucxx://")
