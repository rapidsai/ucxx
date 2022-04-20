# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import glob
import os
import re
from distutils.sysconfig import get_config_var, get_python_inc

from Cython.Build import cythonize
from Cython.Distutils.build_ext import new_build_ext as build_ext
from setuptools import find_packages, setup
from setuptools.extension import Extension

import numpy as np

import versioneer

with open("../README.md", "r") as fh:
    readme = fh.read()

setup_dir = os.path.dirname(os.path.realpath(__file__))
include_dirs = [
    os.path.dirname(get_python_inc()),
    np.get_include(),
    "/usr/local/cuda/include"
]
library_dirs = [get_config_var("LIBDIR"), "/usr/local/cuda/lib64"]
libraries = ["ucp", "uct", "ucm", "ucs", "cudart", "cuda"]
cpp_extra_compile_args = ["-std=c++17", "-Werror", "-g", "-DUCXX_ENABLE_PYTHON"]
c_extra_compile_args = ["-Werror", "-g"]
depends = []

# Add ucxx headers from the source tree (if available)
ucxx_include_dir = os.path.abspath(f"{setup_dir}/../cpp/include")
if os.path.isdir(ucxx_include_dir):
    include_dirs = include_dirs + [ucxx_include_dir]
    depends.extend(glob.glob(f"{ucxx_include_dir}/ucxx/*"))


def get_ucp_version():
    for inc_dir in include_dirs:
        with open(inc_dir + "/ucp/api/ucp_version.h") as f:
            ftext = f.read()
            major = re.findall("^#define.*UCP_API_MAJOR.*", ftext, re.MULTILINE)
            minor = re.findall("^#define.*UCP_API_MINOR.*", ftext, re.MULTILINE)

            major = int(major[0].split()[-1])
            minor = int(minor[0].split()[-1])

            return (major, minor)


_am_supported = 1 if (get_ucp_version() >= (1, 11)) else 0


ext_modules = cythonize(
    [
        Extension(
            "ucxx._lib.libucxx",
            sources=["ucxx/_lib/libucxx.pyx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            depends=depends,
            language="c++",
            extra_compile_args=cpp_extra_compile_args,
        ),
        Extension(
            "ucxx._lib.arr",
            sources=["ucxx/_lib/arr.pyx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=c_extra_compile_args,
        ),
    ],
    compile_time_env={"CY_UCP_AM_SUPPORTED": _am_supported},
)

cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["build_ext"] = build_ext

install_requires = [
    "numpy",
    "pynvml",
]

tests_require = [
    "pytest",
    "pytest-asyncio",
]

setup(
    name="ucxx",
    packages=find_packages(exclude=["tests*"]),
    package_data={"": ["*.pyi"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    version=versioneer.get_version(),
    python_requires=">=3.6",
    install_requires=install_requires,
    tests_require=tests_require,
    description="Python Bindings for the Unified Communication X library (UCX)",
    long_description=readme,
    author="NVIDIA Corporation",
    license="BSD-3-Clause",
    license_files=["LICENSE"],
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware",
        "Topic :: System :: Systems Administration",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    url="https://github.com/rapidsai/ucx-py",
)
