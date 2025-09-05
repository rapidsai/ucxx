# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

# TODO: Remove UCXXConnect and UCXXListener once rapids-dask-dependency doesn't
# need them anymore. `UCXXBackend*` need to remain for `pyproject.toml`.
from .ucxx import (
    UCXXBackend,
    UCXXBackendLegacyPrefix,
    UCXXConnector,
    UCXXListener,
)  # noqa: F401


from ._version import __git_commit__, __version__
