# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from __future__ import absolute_import, print_function

import os
import re
from distutils.sysconfig import get_config_var, get_python_inc

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils.build_ext import new_build_ext
from setuptools import setup
from setuptools.extension import Extension

include_dirs = [
    os.path.dirname(get_python_inc()),
    np.get_include(),
    "/usr/local/cuda/include",
]
library_dirs = [get_config_var("LIBDIR")]
libraries = ["ucxx", "ucxx_python", "fmt"]
cpp_extra_compile_args = [
    "-std=c++17",
    "-Werror",
    "-DUCXX_ENABLE_PYTHON=1",
    "-DUCXX_ENABLE_RMM=1",
]
c_extra_compile_args = ["-Werror"]
depends = []


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

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": new_build_ext},
)
