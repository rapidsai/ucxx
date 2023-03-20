# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from setuptools import find_packages
from skbuild import setup

setup(
    include_package_data=True,
    packages=find_packages(include=["ucxx", "ucxx.*"]),
    zip_safe=False,
)
