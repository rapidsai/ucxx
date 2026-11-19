#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Sourced by the CI entry points in this directory. UCXX is maintained in the
# UCX repository under bindings/ucxx; the entry points fetch that tree and run
# the CI script of the same name from it. See ci/README.md.
#
# Env (all optional):
#   UCXX_UCX_REPOSITORY  UCX repository to fetch (default: openucx/ucx)
#   UCXX_UCX_REF         ref to fetch from it (default: the upstreaming PR)
#   UCXX_UCX_SRC_DIR     UCX checkout to use; fetched unless it already holds
#                        a git checkout, which is then used as is

UCXX_UCX_REPOSITORY="${UCXX_UCX_REPOSITORY:-https://github.com/openucx/ucx}"
# TODO: switch to 'master' once https://github.com/openucx/ucx/pull/12364 is merged.
UCXX_UCX_REF="${UCXX_UCX_REF:-pull/12364/head}"
UCXX_UCX_SRC_DIR="${UCXX_UCX_SRC_DIR:-${TMPDIR:-/tmp}/ucx-src}"

# Root of this (ucxx) repository.
UCXX_REPO_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.." && pwd)"

# Whether UCXX_UCX_SRC_DIR is the top of a git checkout with a commit, not just
# a directory inside another repository or one left over by a failed fetch.
ucx_bindings_has_checkout() {
  [[ -d "${UCXX_UCX_SRC_DIR}" ]] &&
    [[ "$(git -C "${UCXX_UCX_SRC_DIR}" rev-parse --show-toplevel 2> /dev/null)" == \
       "$(realpath "${UCXX_UCX_SRC_DIR}")" ]] &&
    git -C "${UCXX_UCX_SRC_DIR}" rev-parse -q --verify HEAD > /dev/null
}

ucx_bindings_fetch() {
  if ucx_bindings_has_checkout; then
    echo "Using the existing UCX checkout at ${UCXX_UCX_SRC_DIR}"
  else
    echo "Fetching ${UCXX_UCX_REF} from ${UCXX_UCX_REPOSITORY} into ${UCXX_UCX_SRC_DIR}"
    git init -q "${UCXX_UCX_SRC_DIR}" || return 1
    local attempt
    for attempt in 1 2 3 4 5; do
      git -C "${UCXX_UCX_SRC_DIR}" fetch -q --depth 1 \
          "${UCXX_UCX_REPOSITORY}" "${UCXX_UCX_REF}" && break
      if [[ "${attempt}" == 5 ]]; then
        echo "ERROR: failed to fetch ${UCXX_UCX_REF} from ${UCXX_UCX_REPOSITORY}" >&2
        return 1
      fi
      sleep $((attempt * 10))
    done
    git -C "${UCXX_UCX_SRC_DIR}" -c advice.detachedHead=false checkout -q FETCH_HEAD || return 1
  fi
  echo "UCX $(git -C "${UCXX_UCX_SRC_DIR}" rev-parse HEAD) (${UCXX_UCX_REPOSITORY} ${UCXX_UCX_REF})"

  UCXX_SRC_DIR="${UCXX_UCX_SRC_DIR}/bindings/ucxx"
  if [[ ! -f "${UCXX_SRC_DIR}/VERSION" ]]; then
    echo "ERROR: no UCXX tree at ${UCXX_SRC_DIR}" >&2
    return 1
  fi

  # Package versions come from this repository's tags (below), dependency pins
  # from the fetched tree: they must describe the same release.
  local repo_version src_version
  repo_version="$(head -n 1 "${UCXX_REPO_DIR}/VERSION")"
  src_version="$(head -n 1 "${UCXX_SRC_DIR}/VERSION")"
  if [[ "${repo_version}" != "${src_version}" ]]; then
    echo "ERROR: VERSION ${repo_version} of this repository does not match VERSION ${src_version} of ${UCXX_SRC_DIR}" >&2
    return 1
  fi
}

# rapids-generate-version derives the version from the tags of the repository
# it runs in, and the UCX repository carries UCX tags, not UCXX ones. Run it
# in this repository instead.
ucx_bindings_shim_version() {
  local real shim_dir
  real="$(command -v rapids-generate-version)" || return 0
  shim_dir="$(mktemp -d)"
  cat > "${shim_dir}/rapids-generate-version" <<EOF
#!/bin/bash
cd "${UCXX_REPO_DIR}" && exec "${real}" "\$@"
EOF
  chmod +x "${shim_dir}/rapids-generate-version"
  export PATH="${shim_dir}:${PATH}"
}

# Output directories are used after the scripts return, from the workspace:
# keep them valid across the change of directory.
for _ucx_bindings_var in RAPIDS_CONDA_BLD_OUTPUT_DIR RAPIDS_WHEEL_BLD_OUTPUT_DIR RAPIDS_TESTS_DIR; do
  if [[ -n "${!_ucx_bindings_var:-}" && "${!_ucx_bindings_var}" != /* ]]; then
    export "${_ucx_bindings_var}=${PWD}/${!_ucx_bindings_var}"
  fi
done
unset _ucx_bindings_var

ucx_bindings_fetch || exit 1
ucx_bindings_shim_version
cd "${UCXX_SRC_DIR}" || exit 1
