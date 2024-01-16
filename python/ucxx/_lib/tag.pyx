# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


from libc.stdint cimport uint64_t


cdef class UCXXTag:
    cdef:
        readonly uint64_t value

    def __cinit__(self, uint64_t tag) -> None:
        self.value = tag


cdef class UCXXTagMask:
    cdef:
        readonly uint64_t value

    def __cinit__(self, uint64_t tag_mask) -> None:
        self.value = tag_mask


UCXXTagMaskFull = UCXXTagMask(2 ** 64 - 1)
