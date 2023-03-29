# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages
from skbuild import setup

packages = find_packages(include=["ucxx*"])
setup(
    packages=packages,
    package_data={key: ["*.pxd"] for key in packages},
    zip_safe=False,
)
