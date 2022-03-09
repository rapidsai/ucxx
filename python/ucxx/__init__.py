# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import logging

from . import testing  # noqa
from . import _version
from ._lib import libucxx  # type: ignore
from .core import *  # noqa
from .utils import get_address, get_ucxpy_logger  # noqa

logger = logging.getLogger("ucx")

# Setup UCX-Py logger
logger = get_ucxpy_logger()

__version__ = _version.get_versions()["version"]
