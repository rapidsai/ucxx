# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

class Array:
    def __init__(self, obj: object): ...
    @property
    def c_contiguous(self) -> bool: ...
    @property
    def f_contiguous(self) -> bool: ...
    @property
    def contiguous(self) -> bool: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def shape(self) -> Tuple[int]: ...
    @property
    def strides(self) -> Tuple[int]: ...
