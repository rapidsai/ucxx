# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from .ucxx import (
    UCXXBackend as UCXXBackend,
    UCXXBackendLegacyPrefix as UCXXBackendLegacyPrefix,
)


from ._version import __git_commit__ as __git_commit__, __version__ as __version__
