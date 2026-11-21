# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from ucxx.benchmarks.utils import get_cuda_visible_device


def test_cuda_visible_device_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert get_cuda_visible_device(3) == "3"


@pytest.mark.parametrize(
    "visible,index,expected",
    [
        ("7", 0, "7"),
        ("6,7", 1, "7"),
        ("10, 11", 1, "11"),
        ("GPU-aaaa,GPU-bbbb", 1, "GPU-bbbb"),
    ],
)
def test_cuda_visible_device_maps_index(monkeypatch, visible, index, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert get_cuda_visible_device(index) == expected


@pytest.mark.parametrize("visible,index", [("7", 1), ("6,7", 6), ("", 0)])
def test_cuda_visible_device_out_of_range(monkeypatch, visible, index):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="is not visible"):
        get_cuda_visible_device(index)
