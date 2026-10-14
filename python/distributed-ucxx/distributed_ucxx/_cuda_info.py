# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA device and context queries implemented with ``cuda.core.system``."""

from __future__ import annotations

import os
from typing import NamedTuple

from cuda import pathfinder
from cuda.core import system

# Errors that indicate NVML (or the driver) isn't usable on this machine.
_NVML_UNAVAILABLE_ERRORS = (
    pathfinder.DynamicLibNotFoundError,
    system.LibraryNotFoundError,
    system.DriverNotLoadedError,
    system.UnknownError,
)


class CudaDeviceInfo(NamedTuple):
    uuid: str | None = None
    device_index: int | None = None
    mig_index: int | None = None


class CudaContext(NamedTuple):
    has_context: bool
    device_info: CudaDeviceInfo | None = None


def device_get_count() -> int:
    """Number of NVML-visible devices, or 0 if NVML is unavailable."""
    try:
        return system.Device.get_device_count()
    except _NVML_UNAVAILABLE_ERRORS:
        return 0


def _get_device(device: int | bytes | str) -> system.Device:
    try:
        return system.Device(index=int(device))
    except ValueError:
        return system.Device(uuid=device)


def _running_process_matches(device: system.Device) -> bool:
    pid = os.getpid()
    return any(proc.pid == pid for proc in device.compute_running_processes)


def _is_mig_enabled(device: system.Device) -> bool:
    try:
        return device.mig.mode
    except system.NotSupportedError:
        return False


def has_cuda_context() -> CudaContext:
    """Check whether the current process already has a CUDA context created.

    Returns
    -------
    out : CudaContext
        Whether the current process has a CUDA context, and if so, information
        about the device the context belongs to.
    """
    try:
        devices = list(system.Device.get_all_devices())
    except _NVML_UNAVAILABLE_ERRORS:
        return CudaContext(has_context=False)

    for index, device in enumerate(devices):
        if _is_mig_enabled(device):
            for mig_index in range(device.mig.device_count):
                try:
                    mig_device = device.mig.get_device_by_index(mig_index)
                except system.NotFoundError:
                    # No MIG device with that index
                    continue
                if _running_process_matches(mig_device):
                    return CudaContext(
                        has_context=True,
                        device_info=CudaDeviceInfo(
                            uuid=mig_device.uuid,
                            device_index=index,
                            mig_index=mig_index,
                        ),
                    )
        elif _running_process_matches(device):
            return CudaContext(
                has_context=True,
                device_info=CudaDeviceInfo(uuid=device.uuid, device_index=index),
            )
    return CudaContext(has_context=False)


def get_device_index_and_uuid(device: int | bytes | str) -> CudaDeviceInfo:
    """Get both device index and UUID from a device index or UUID.

    Parameters
    ----------
    device : int, bytes or str
        An ``int`` with the index of a GPU, or ``bytes`` or ``str`` with the UUID
        of a CUDA (either GPU or MIG) device.

    Returns
    -------
    out : CudaDeviceInfo
        Object containing information about the device.
    """
    dev = _get_device(device)
    return CudaDeviceInfo(uuid=dev.uuid, device_index=dev.index)


def get_device_mig_mode(device: int | bytes | str) -> list[int]:
    """Get MIG mode for a device index or UUID.

    Returns
    -------
    out : list
        A ``list`` with two integers ``[current_mode, pending_mode]``, where
        ``1`` means enabled and ``0`` disabled.
    """
    dev = _get_device(device)
    try:
        return [int(dev.mig.mode), int(dev.mig.pending_mode)]
    except system.NotSupportedError:
        return [0, 0]
