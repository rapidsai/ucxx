# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from ucxx._lib.libucxx import get_ucx_version

UCX_LT_1_22_0 = get_ucx_version() < (1, 22, 0)

__all__ = ["UCX_LT_1_22_0"]
