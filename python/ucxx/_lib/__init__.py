# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from .libucxx import _create_exceptions

# Ensure Python exceptions are created before use
_create_exceptions()
