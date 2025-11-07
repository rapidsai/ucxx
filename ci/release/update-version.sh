#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

########################
# UCXX Version Updater #
########################

## Usage
# NOTE: This script must be run from the repository root, not from the ci/release/ directory
# Primary interface:   bash ci/release/update-version.sh <new_version> [--run-context=main|release]
# Fallback interface:  [RAPIDS_RUN_CONTEXT=main|release] bash ci/release/update-version.sh <new_version>
# CLI arguments take precedence over environment variables
# Defaults to main when no run-context is specified


# Parse command line arguments
CLI_RUN_CONTEXT=""
VERSION_ARG=""

for arg in "$@"; do
    case ${arg} in
        --run-context=*)
            CLI_RUN_CONTEXT="${arg#*=}"
            shift
            ;;
        *)
            if [[ -z "${VERSION_ARG}" ]]; then
                VERSION_ARG="${arg}"
            fi
            ;;
    esac
done

# Format is Major.Minor.Patch - no leading 'v' or trailing 'a'
# Example: 0.30.00
NEXT_FULL_TAG="${VERSION_ARG}"

# Determine RUN_CONTEXT with CLI precedence over environment variable, defaulting to main
if [[ -n "${CLI_RUN_CONTEXT}" ]]; then
    RUN_CONTEXT="${CLI_RUN_CONTEXT}"
    echo "Using run-context from CLI: ${RUN_CONTEXT}"
elif [[ -n "${RAPIDS_RUN_CONTEXT}" ]]; then
    RUN_CONTEXT="${RAPIDS_RUN_CONTEXT}"
    echo "Using run-context from environment: ${RUN_CONTEXT}"
else
    RUN_CONTEXT="main"
    echo "No run-context provided, defaulting to: ${RUN_CONTEXT}"
fi

# Validate RUN_CONTEXT value
if [[ "${RUN_CONTEXT}" != "main" && "${RUN_CONTEXT}" != "release" ]]; then
    echo "Error: Invalid run-context value '${RUN_CONTEXT}'"
    echo "Valid values: main, release"
    exit 1
fi

# Validate version argument
if [[ -z "${NEXT_FULL_TAG}" ]]; then
    echo "Error: Version argument is required"
    echo "Usage: ${0} <new_version> [--run-context=<context>]"
    echo "   or: [RAPIDS_RUN_CONTEXT=<context>] ${0} <new_version>"
    echo "Note: Defaults to main when run-context is not specified"
    exit 1
fi

# Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "${NEXT_FULL_TAG}" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "${NEXT_FULL_TAG}" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Get RAPIDS version associated w/ ucx-py version
NEXT_RAPIDS_SHORT_TAG="$(curl -sL "https://version.gpuci.io/ucx-py/${NEXT_SHORT_TAG}")"

# Need to distutils-normalize the versions for some use cases
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")
NEXT_RAPIDS_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_RAPIDS_SHORT_TAG}'))")
echo "Next tag is ${NEXT_RAPIDS_SHORT_TAG_PEP440}"

# Set branch references based on RUN_CONTEXT
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    RAPIDS_BRANCH_NAME="main"
    echo "Preparing development branch update => ${NEXT_FULL_TAG} (targeting main branch)"
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    RAPIDS_BRANCH_NAME="release/${NEXT_RAPIDS_SHORT_TAG}"
    echo "Preparing release branch update => ${NEXT_FULL_TAG} (targeting release/${NEXT_RAPIDS_SHORT_TAG} branch)"
fi

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"${1}"'' "${2}" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION
echo "${NEXT_RAPIDS_SHORT_TAG}.00" > RAPIDS_VERSION
echo "${RAPIDS_BRANCH_NAME}" > RAPIDS_BRANCH

# Update RAPIDS version
for FILE in conda/recipes/*/conda_build_config.yaml; do
  sed_runner "/^rapids_version:$/ {n;s/.*/  - \"${NEXT_RAPIDS_SHORT_TAG_PEP440}\.*\"/}" "${FILE}"
done

DEPENDENCIES=(
  cudf
  dask-cuda
  dask-cudf
  librmm
  rapids-dask-dependency
  rmm
)
UCXX_DEPENDENCIES=(
  distributed-ucxx
  libucxx
  libucxx-examples
  libucxx-tests
  ucxx
  ucxx-tests
)
for FILE in dependencies.yaml conda/environments/*.yaml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_RAPIDS_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  for DEP in "${UCXX_DEPENDENCIES[@]}"; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
done
for FILE in python/*/pyproject.toml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/\"${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*\"/==${NEXT_RAPIDS_SHORT_TAG_PEP440}\.*,>=0.0.0a0\"/g" "${FILE}"
  done
  for DEP in "${UCXX_DEPENDENCIES[@]}"; do
    sed_runner "/\"${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}\.*,>=0.0.0a0\"/g" "${FILE}"
  done
done

# Update cmake/RAPIDS.cmake with context-aware branch references
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    sed_runner "s|\"release/\${rapids-cmake-version}\"|\"main\"|g" cmake/RAPIDS.cmake
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    sed_runner "s|\"main\"|\"release/\${rapids-cmake-version}\"|g" cmake/RAPIDS.cmake
fi

# Documentation references - context-aware
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    # In main context, use main branch for documentation links
    sed_runner "s|/blob/branch-[^/]*/|/blob/main/|g" docs/ucxx/source/send_recv.rst
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    # In release context, use release branch for documentation links
    sed_runner "s|/blob/main/|/blob/release/${NEXT_SHORT_TAG}/|g" docs/ucxx/source/send_recv.rst
    sed_runner "s|/blob/branch-[^/]*/|/blob/release/${NEXT_SHORT_TAG}/|g" docs/ucxx/source/send_recv.rst
fi

# CI files - context-aware branch references
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s|@.*|@${RAPIDS_BRANCH_NAME}|g" "${FILE}"
  sed_runner "s|:[0-9]*\\.[0-9]*-|:${NEXT_RAPIDS_SHORT_TAG}-|g" "${FILE}"
done

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_RAPIDS_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_RAPIDS_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-[0-9.]*@rapids-\${localWorkspaceFolderBasename}-${NEXT_RAPIDS_SHORT_TAG}@g" "${filename}"
done
