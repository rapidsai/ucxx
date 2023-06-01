#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

# UCXX build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)
# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

VALIDARGS="clean libucxx libucxx_python ucxx benchmarks tests examples -v -g -n -l --show_depr_warn -h"
HELP="$0 [clean] [libucxx] [libucxx_python] [ucxx] [benchmarks] [tests] [examples] [-v] [-g] [-n] [-h] [--cmake-args=\\\"<args>\\\"]
   clean                         - remove all existing build artifacts and configuration (start
                                   over)
   libucxx                       - build the UCXX C++ module
   libucxx_python                - build the UCXX C++ Python support module
   ucxx                          - build the ucxx Python package
   benchmarks                    - build benchmarks
   tests                         - build tests
   examples                      - build examples
   -v                            - verbose build mode
   -g                            - build for debug
   -n                            - no install step
   --show_depr_warn              - show cmake deprecation warnings
   --cmake-args=\\\"<args>\\\"   - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   -h | --h[elp]                 - print this text

   default action (no args) is to build and install 'libucxx' and 'libucxx_python', and then 'ucxx' targets
"
LIB_BUILD_DIR=${LIB_BUILD_DIR:=${REPODIR}/cpp/build}
UCXX_BUILD_DIR=${REPODIR}/python/build

BUILD_DIRS="${LIB_BUILD_DIR} ${UCXX_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_BENCHMARKS=OFF
BUILD_TESTS=OFF
BUILD_EXAMPLES=OFF
BUILD_DISABLE_DEPRECATION_WARNINGS=ON
UCXX_ENABLE_PYTHON=OFF
UCXX_ENABLE_RMM=OFF

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo $ARGS | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo $ARGS | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo $ARGS | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo $EXTRA_CMAKE_ARGS | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi
}

function buildAll {
    ((${NUMARGS} == 0 )) || !(echo " ${ARGS} " | grep -q " [^-]\+ ")
}

if hasArg -h || hasArg --h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    # Check for cmake args
    cmakeArgs
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option or formatting, check --help: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG="-v"
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
    LIBUCXX_BUILD_DIR=${LIB_BUILD_DIR}
else
    LIBUCXX_BUILD_DIR=${CONDA_PREFIX}/lib
fi
if hasArg benchmarks; then
    BUILD_BENCHMARKS=ON
fi
if hasArg tests; then
    BUILD_TESTS=ON
fi
if hasArg examples; then
    BUILD_EXAMPLES=ON
fi
if hasArg --show_depr_warn; then
    BUILD_DISABLE_DEPRECATION_WARNINGS=OFF
fi

if buildAll || hasArg libucxx_python; then
  UCXX_ENABLE_PYTHON=ON
  UCXX_ENABLE_RMM=ON
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
    if [ -d ${bd} ]; then
        find ${bd} -mindepth 1 -delete
        rmdir ${bd} || true
    fi
    done

    # Cleaning up python artifacts
    find ${REPODIR}/python/ | grep -E "(__pycache__|\.pyc|\.pyo|\.so|\_skbuild$)"  | xargs rm -rf

fi


################################################################################
# Configure, build, and install libucxxx


if buildAll || hasArg libucxx; then
    pwd
    cmake -S $REPODIR/cpp -B ${LIB_BUILD_DIR} \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DBUILD_BENCHMARKS=${BUILD_BENCHMARKS} \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DBUILD_EXAMPLES=${BUILD_EXAMPLES} \
          -DDISABLE_DEPRECATION_WARNINGS=${BUILD_DISABLE_DEPRECATION_WARNINGS} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DUCXX_ENABLE_PYTHON=${UCXX_ENABLE_PYTHON} \
          -DUCXX_ENABLE_RMM=${UCXX_ENABLE_RMM} \
          ${EXTRA_CMAKE_ARGS}

    cd ${LIB_BUILD_DIR}

    compile_start=$(date +%s)
    cmake --build . -j${PARALLEL_LEVEL} ${VERBOSE_FLAG}
    compile_end=$(date +%s)
    compile_total=$(( compile_end - compile_start ))

    if [[ ${INSTALL_TARGET} != "" ]]; then
        cmake --build . -j${PARALLEL_LEVEL} --target install ${VERBOSE_FLAG}
        if [[ ${UCXX_ENABLE_PYTHON} == "ON" ]]; then
          cmake --install . --component python
        fi
        if [[ ${BUILD_BENCHMARKS} == "ON" ]]; then
          cmake --install . --component benchmarks
        fi
        if [[ ${BUILD_EXAMPLES} == "ON" ]]; then
          cmake --install . --component examples
        fi
        if [[ ${BUILD_TESTS} == "ON" ]]; then
          cmake --install . --component testing
        fi
    fi
fi

# Append `-DFIND_UCXX_CPP=ON` to EXTRA_CMAKE_ARGS unless a user specified the option.
if [[ "${EXTRA_CMAKE_ARGS}" != *"DFIND_UCXX_CPP"* ]]; then
    EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DFIND_UCXX_CPP=ON"
fi

# Build and install the UCXX Python package
if buildAll || hasArg ucxx; then

    cd ${REPODIR}/python/
    python setup.py build_ext --inplace -- -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_LIBRARY_PATH=${LIBUCXX_BUILD_DIR} -DCMAKE_CUDA_ARCHITECTURES=${UCXX_CMAKE_CUDA_ARCHITECTURES} ${EXTRA_CMAKE_ARGS} -- -j${PARALLEL_LEVEL:-1}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        python setup.py install --single-version-externally-managed --record=record.txt  -- -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_LIBRARY_PATH=${LIBUCXX_BUILD_DIR} ${EXTRA_CMAKE_ARGS} -- -j${PARALLEL_LEVEL:-1}
    fi
fi
