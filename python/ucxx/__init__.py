# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


from . import _version
from . import testing  # noqa
from ._lib import libucxx  # type: ignore

__version__ = _version.get_versions()["version"]


class UCXContext:
    def __init__(self, config, feature_flags):
        self._handle = libucxx.UCXContext(config, feature_flags)
