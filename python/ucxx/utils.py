# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import asyncio
import fcntl
import glob
import hashlib
import logging
import multiprocessing as mp
import os
import socket
import struct
import time

import numpy as np

from ._lib import libucxx as ucx_api

mp = mp.get_context("spawn")


def get_ucxpy_logger():
    """
    Get UCX-Py logger with custom formatting

    Returns
    -------
    logger : logging.Logger
        Logger object

    Examples
    --------
    >>> logger = get_ucxpy_logger()
    >>> logger.warning("Test")
    [1585175070.2911468] [dgx12:1054] UCXPY  WARNING Test
    """

    _level_enum = logging.getLevelName(os.getenv("UCXPY_LOG_LEVEL", "WARNING"))
    logger = logging.getLogger("ucx")

    # Avoid duplicate logging
    logger.propagate = False

    class LoggingFilter(logging.Filter):
        def filter(self, record):
            record.hostname = socket.gethostname()
            record.timestamp = str("%.6f" % time.time())
            return True

    formatter = logging.Formatter(
        "[%(timestamp)s] [%(hostname)s:%(process)d] UCXPY  %(levelname)s %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(LoggingFilter())
    logger.addHandler(handler)

    logger.setLevel(_level_enum)

    return logger


def hash64bits(*args):
    """64 bit unsigned hash of `args`"""
    # 64 bits hexdigest
    h = hashlib.sha1(bytes(repr(args), "utf-8")).hexdigest()[:16]
    # Convert to an integer and return
    return int(h, 16)


def get_address(ifname=None):
    """
    Get the address associated with a network interface.

    Parameters
    ----------
    ifname : str
        The network interface name to find the address for.
        If None, it uses the value of environment variable `UCXPY_IFNAME`
        and if `UCXPY_IFNAME` is not set it defaults to "ib0"
        An OSError is raised for invalid interfaces.

    Returns
    -------
    address : str
        The inet addr associated with an interface.

    Examples
    --------
    >>> get_address()
    '10.33.225.160'

    >>> get_address(ifname='lo')
    '127.0.0.1'
    """

    def _get_address(ifname):
        ifname = ifname.encode()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            return socket.inet_ntoa(
                fcntl.ioctl(
                    s.fileno(), 0x8915, struct.pack("256s", ifname[:15])  # SIOCGIFADDR
                )[20:24]
            )

    def _try_interfaces():
        prefix_priority = ["ib", "eth", "en"]
        iftypes = {p: [] for p in prefix_priority}
        for i in glob.glob("/sys/class/net/*"):
            name = i.split("/")[-1]
            for p in prefix_priority:
                if name.startswith(p):
                    iftypes[p].append(name)
        for p in prefix_priority:
            iftype = iftypes[p]
            iftype.sort()
            for i in iftype:
                try:
                    return _get_address(i)
                except OSError:
                    pass

    if ifname is None:
        ifname = os.environ.get("UCXPY_IFNAME")

    if ifname is not None:
        return _get_address(ifname)
    else:
        return _try_interfaces()
