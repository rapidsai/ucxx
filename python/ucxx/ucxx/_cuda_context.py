# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA context management using cuda.core.

Provides helpers to ensure a CUDA context is created and to synchronize
the default stream.
"""


def _get_device_class():
    """Get the Device class from cuda.core."""
    try:
        from cuda.core import Device

        return Device
    except ImportError:
        try:
            from cuda.core.experimental import Device

            return Device
        except ImportError as e:
            raise ImportError(
                "CUDA context management requires cuda-core (cuda-core>=0.3.2)."
            ) from e


def ensure_cuda_context(device_id: int = 0) -> None:
    """Ensure a CUDA context exists for the given device and set it as current.

    Parameters
    ----------
    device_id : int, optional
        The CUDA device index (default: 0).
    """
    Device = _get_device_class()
    Device(device_id).set_current()


def synchronize_default_stream(device_id: int = 0) -> None:
    """Synchronize the default CUDA stream of the current device.

    Required when coordinating with UCX CUDA transfers (e.g. before send/recv
    of CUDA buffers).

    Parameters
    ----------
    device_id : int, optional
        The CUDA device index (default: 0).
    """
    Device = _get_device_class()
    device = Device(device_id)
    device.set_current()
    device.sync()
