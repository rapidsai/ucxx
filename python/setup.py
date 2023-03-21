# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from setuptools import find_packages
from skbuild import setup

packages = find_packages(include=["ucxx*"])
setup(
    packages=packages,
    package_data={key: ["*.pxd"] for key in packages},
    zip_safe=False,
)
