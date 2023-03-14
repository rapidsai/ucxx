# UCXX Conda build Docker container

## Summary

Contains dockerfile and script to build UCXX conda packages.

## Building Docker image

Before starting the build process, it's necessary to build the image, this is done as follows:

```bash
docker build -t ucxx-conda-build -f conda-build.dockerfile .
```

### Controlling build-args

You can control some of the behaviour of the docker file with docker `--build-arg` flags:

- `DISTRIBUTION_VERSION`: version of distribution in the base image (default `ubuntu20.04`), must exist in the [nvidia/cuda](https://hub.docker.com/layers/cuda/nvidia/cuda) docker hub image list;
- `CUDA_VERSION`: version of cuda toolkit in the base image (default `11.8.0`), must exist in the [nvidia/cuda](https://hub.docker.com/layers/cuda/nvidia/cuda) docker hub image list;
- `CONDA_HOME`: Where to install conda in the image (default `/opt/conda`);

## Building Conda packages

Once the Docker image is built, Conda packages can be built by running the container as below:

```bash
docker run -v /path/to/src/ucxx:/ucxx ucxx-conda-build
```

The process above may take various minutes depending on the system where this is running. Once it completes, the Docker container will exit and the resulting packages will be available in `.conda-bld/linux-64/`. Conda build artifacts will remain in `.conda-bld/` and may be deleted if you so choose.

## Installing Conda packages

To install the resulting Conda packages it is only required that you have the Conda environment where it should be installed activated and run:

```bash
conda install -c .conda-bld/linux-64 -c conda-forge ucxx
```

The command above will install the `ucxx` package with the UCXX Python implementation as well as its dependencies. Installing the C++ package only can be achieved by replacing `ucxx` by `libucxx`:

```bash
conda install -c .conda-bld/linux-64 -c conda-forge libucxx
```

### Build parameters and dependencies

The following arguments may be specified to control build parameters and dependencies:

- `GIT_DESCRIBE_NUMBER` (default: 0): 
- `PYTHON_VERSION` (default: 3.10): Python version to build for;
- `NUMPY_VERSION` (default: 1.21): NumPy version to build against;
- `RAPIDS_DATE_STRING` (default: `date +%y%m%d`): Date string to use for RAPIDS package build;
- `RAPIDS_VERSION` (default: 23.04): RAPIDS version to build against (RMM dependency);
- `RAPIDS_CUDA_VERSION` (default: 11.8): CUDA version to build for (must match Docker image's major CUDA version);
- `PYNVML_MIN_VERSION` (default: 11.4.1): Minimum PyNVML version required to run;

For example:

```bash
docker run -v /path/to/src/ucxx:/ucxx \
  -e PYTHON_VERSION="3.8" \
  -e RAPIDS_CUDA_VERSION="11.7" \
  ucxx-conda-build
```
