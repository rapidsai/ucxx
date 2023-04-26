#!/bin/bash
########################
# UCXX Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


# Format is Major.Minor.Patch - no leading 'v' or trailing 'a'
# Example: 0.30.00
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag | grep -xE 'v[0-9\.]+' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Get RAPIDS version associated w/ ucx-py version
CURRENT_RAPIDS_VERSION="$(curl -sL https://version.gpuci.io/ucx-py/${CURRENT_SHORT_TAG})"
NEXT_RAPIDS_VERSION="$(curl -sL https://version.gpuci.io/ucx-py/${NEXT_SHORT_TAG})"

# Need to distutils-normalize the versions for some use cases
CURRENT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${CURRENT_RAPIDS_VERSION}'))")
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_RAPIDS_VERSION}'))")
echo "Current RAPIDS is ${CURRENT_SHORT_TAG_PEP440}, next is ${NEXT_SHORT_TAG_PEP440}"

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# bump cudf, dask-cuda and rmm
sed_runner "s/rmm =${CURRENT_RAPIDS_VERSION}/rmm =${NEXT_RAPIDS_VERSION}/g" conda/recipes/ucxx/meta.yaml
for FILE in conda/environments/*.yaml dependencies.yaml; do
  sed_runner "s/cudf==${CURRENT_SHORT_TAG_PEP440}/cudf==${NEXT_SHORT_TAG_PEP440}/g" ${FILE};
  sed_runner "s/dask-cuda==${CURRENT_SHORT_TAG_PEP440}/dask-cuda==${NEXT_SHORT_TAG_PEP440}/g" ${FILE};
  sed_runner "s/rmm==${CURRENT_SHORT_TAG_PEP440}/rmm==${NEXT_SHORT_TAG_PEP440}/g" ${FILE};
done

# rapids-cmake version
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_RAPIDS_VERSION}\/RAPIDS.cmake"'/g' fetch_rapids.cmake

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-action-workflows/ s/@.*/@branch-${NEXT_RAPIDS_VERSION}/g" "${FILE}"
done
