# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

setup(
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests.*"]),
    entry_points={"distributed.comm.backends": ["ucxx=distributed_ucxx:UCXXBackend"]},
    zip_safe=False,
)
