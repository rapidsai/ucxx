#!/bin/bash
########################
# UCXX Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


# Format is Major.Minor.Patch - no leading 'v' or trailing 'a'
# Example: 0.30.00
NEXT_FULL_TAG=$1

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Get RAPIDS version associated w/ ucx-py version
NEXT_RAPIDS_VERSION="$(curl -sL https://version.gpuci.io/ucx-py/${NEXT_SHORT_TAG})"

# Need to distutils-normalize the versions for some use cases
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")
NEXT_RAPIDS_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_RAPIDS_VERSION}'))")
echo "Next tag is ${NEXT_RAPIDS_SHORT_TAG_PEP440}"

echo "Preparing release: $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION

# bump RAPIDS libs
sed_runner "/^rapids_version:$/ {n;s/.*/  - \"${NEXT_RAPIDS_SHORT_TAG_PEP440}\.*\"/}" conda/recipes/ucxx/conda_build_config.yaml

DEPENDENCIES=(
  cudf
  dask-cuda
  dask-cudf
  librmm
  rmm
  rapids-dask-dependency
)
for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_RAPIDS_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_RAPIDS_SHORT_TAG_PEP440}\.*,>=0.0.0a0\"/g" python/pyproject.toml
  sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_RAPIDS_SHORT_TAG_PEP440}\.*,>=0.0.0a0\"/g" python/distributed-ucxx/pyproject.toml
done

UCXX_DEPENDENCIES=(
  ucxx
  distributed-ucxx
)
for DEP in "${UCXX_DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*/,>=0.0.0a0/g" "${FILE}"
  done
  sed_runner "/\"${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}\.*,>=0.0.0a0\"/g" python/pyproject.toml
  sed_runner "/\"${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}\.*,>=0.0.0a0\"/g" python/distributed-ucxx/pyproject.toml
done

# rapids-cmake version
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_RAPIDS_VERSION}\/RAPIDS.cmake"'/g' fetch_rapids.cmake

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_RAPIDS_VERSION}/g" "${FILE}"
done

sed_runner "s/--rapids-version=[[:digit:]]\{2\}.[[:digit:]]\{2\}/--rapids-version=${NEXT_RAPIDS_VERSION}/g" .pre-commit-config.yaml
