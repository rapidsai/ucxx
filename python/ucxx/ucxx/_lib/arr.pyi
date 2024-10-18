# SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Generic, Tuple, TypeVar

T = TypeVar("T")

class Array(Generic[T]):
    def __init__(self, obj: T): ...
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
    @property
    def cuda(self) -> bool: ...
    @property
    def obj(self) -> T: ...

def asarray(obj) -> Array: ...
