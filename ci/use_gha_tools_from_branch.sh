#!/bin/bash

export RAPIDS_CONDA_RETRY_TIMEOUT=120

# fill these in
GHA_TOOLS_BRANCH='conda-install-timeout'
GHA_TOOLS_REPO_ORG=jameslamb

git clone \
  --branch ${GHA_TOOLS_BRANCH} \
  https://github.com/${GHA_TOOLS_REPO_ORG}/gha-tools.git \
  /tmp/gha-tools

unset GHA_TOOLS_BRANCH GHA_TOOLS_REPO_ORG

export PATH="/tmp/gha-tools/tools":$PATH
